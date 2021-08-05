import os
import sys
import argparse

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer, seed_everything

import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, AutoConfig, ElectraForMaskedLM, ElectraForPreTraining, ElectraTokenizer
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR


from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import loggers as pl_loggers

from utils.utils import KorELECTRADataset

from tqdm import tqdm, trange
from electra_pytorch import Electra
import pickle

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return optimizer_grouped_parameters


def tie_weights(generator, discriminator):
    generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings


class LogitsAdapter(torch.nn.Module):
    def __init__(self, adaptee):
        super().__init__()
        self.adaptee = adaptee

    def forward(self, *args, **kwargs):
        return self.adaptee(*args, **kwargs)[0]


def get_train_examples(corpus_path, corpus_lines, on_memory):

    """Load train dataset corpus"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        if corpus_lines is None and not on_memory:
            for _ in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                corpus_lines += 1

        if on_memory:
            lines = [line[:-1].split("\t") for line in tqdm(f, desc="Loading Dataset", total=corpus_lines)]
            corpus_lines = len(lines)
    return lines


class Model(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)  # 이 부분에서 self.hparams에 위 kwargs가 저장된다.

        self.generator = ElectraForMaskedLM(AutoConfig.from_pretrained(self.hparams.model_generator))
        self.discriminator = ElectraForPreTraining(AutoConfig.from_pretrained(self.hparams.model_discriminator))

        tie_weights(self.generator, self.discriminator)

        self.tokenizer = ElectraTokenizer.from_pretrained(self.hparams.vocab_path, do_lower_case=False, add_special_tokens=False)

        self.model = Electra(
            LogitsAdapter(self.generator),
            LogitsAdapter(self.discriminator),
            num_tokens=len(self.tokenizer),
            mask_token_id=self.tokenizer.vocab['[MASK]'],
            pad_token_id=self.tokenizer.vocab['[PAD]'],
            mask_prob=self.hparams.model_mask_prob,
            mask_ignore_token_ids=[self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab['[SEP]'], self.tokenizer.vocab['[PAD]']],
            random_token_prob=0.0)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):

        input_ids, input_mask, segment_ids = batch
        input_ids, input_mask, segment_ids = input_ids.squeeze(), input_mask.squeeze(), segment_ids.squeeze()


        loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = self(input_ids,
                                                                                    attention_mask=input_mask,
                                                                                    token_type_ids=segment_ids)

        del input_ids, input_mask, segment_ids

        self.log('loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_mlm', loss_mlm.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_disc', loss_disc.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_gen', acc_gen.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_disc', acc_disc.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.global_step > 0 and self.global_step % self.hparams.step_ckpt == 0 and self.global_rank == 0:
            self.discriminator.electra.save_pretrained(
                f'{self.hparams.output_path}/ckpt/{self.current_epoch}/{self.global_step}')

        return {
            'loss': loss,
            # 'loss_mlm': loss_mlm,
            # 'loss_disc': loss_disc,

        }

    def configure_optimizers(self):
        print("configure_optimizers")
        optimizer = torch.optim.AdamW(
            get_params_without_weight_decay_ln(self.model.named_parameters(),
                                               weight_decay=self.hparams.adam_weight_decay),
            lr=self.hparams.lr, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2), eps=1e-08)

        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                         num_training_steps=self.hparams.num_training_steps),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1}


        return [optimizer], [lr_scheduler]

    def dataloader(self, train_data_path, shuffle=False):
        print("dataloader")
        # train_examples = get_train_examples(corpus_path=train_data_path, corpus_lines=self.hparams.corpus_lines,
        #                                     on_memory=self.hparams.on_memory)
        train_examples = []
        corpus_names = os.listdir(train_data_path)
        for corpus_name in corpus_names:
            train_examples.extend(get_train_examples(corpus_path=os.path.join(train_data_path, corpus_name),
                                                     corpus_lines=self.hparams.corpus_lines, on_memory=self.hparams.on_memory))

        train_dataset = KorELECTRADataset(train_examples, self.tokenizer, self.hparams)

        del train_examples

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)



def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_path(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = os.path.join('output/' + exp_id, t)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def copy_source(file, output_path):
    import shutil
    shutil.copyfile(file, os.path.join(output_path, os.path.basename(file)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--tpu_cores", type=int, default=8)
    parser.add_argument("--cpu_workers", type=int, default=4)
    parser.add_argument("--model_mask_prob", type=float, default=0.15)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=30_000)
    parser.add_argument("--num_training_steps", type=int, default=1_200_00)
    parser.add_argument("--step_log", type=int, default=10)
    parser.add_argument("--step_ckpt", type=int, default=20_000)

    parser.add_argument("--train_data_path", required=True, type=str)
    parser.add_argument("--vocab_path", required=True, type=str)
    parser.add_argument("--model_generator", required=True, type=str)
    parser.add_argument("--model_discriminator", required=True, type=str)
    parser.add_argument("--dir_ckpt", required=True, type=str)

    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--corpus_lines", type=str, default=None)
    parser.add_argument("--on_memory", type=bool, default=True)
    parser.add_argument("--bool_ckpt", type=bool, default=False)

    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--lr_scheduler", type=str, default='exp')

    # args = parser.parse_args(['--train_data_path', './data/corpus/',
    #                           '--vocab_path', './data/wiki_data_vocab.txt',
    #                           '--model_generator', './small_generator.json',
    #                           '--model_discriminator', './small_discriminator.json',
    #                           '--dir_ckpt', './checkpoint/ckpt'])

    args = parser.parse_args()

    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)

    # preamble
    exp_id = get_exp_id(__file__)
    output_path = get_output_path(exp_id)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/ckpt', exist_ok=False)
    copy_source(__file__, output_path)

    args.output_path = output_path

    model = Model(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_path, 'ckpt'),
        filename='epoch{epoch}-step{step}-loss_step{loss_step:.4f}',
        monitor='loss_step',
        save_last=True,
        save_top_k=3,
        # period=1,
        every_n_train_steps=args.step_ckpt,
        auto_insert_metric_name=False,
    )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(output_path, 'logs/'), name='model', default_hp_metric=False)
    # os.makedirs(f'{output_path}/logs', exist_ok=False)

    print(":: Start Training ::")

    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        precision=16 if args.fp16 else 32,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        move_metrics_to_cpu=True,
        # For TPU Setup
        tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    if os.path.isdir(args.dir_ckpt):
        trainer.resume_from_checkpoint = os.path.join(args.dir_ckpt, os.listdir(args.dir_ckpt)[0])

    trainer.fit(model)


if __name__ == '__main__':
    main()


