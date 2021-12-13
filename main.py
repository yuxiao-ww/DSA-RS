# -*- coding: UTF-8 -*-
import argparse
import random
import time
import sklearn.metrics as metric
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from utils import *
from model import *
from transformers import (
    AdamW,
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['IMDB', 'yelp13', 'yelp14'], default='IMDB')
parser.add_argument('--learning_rate_bert', type=float, default=5e-5)
parser.add_argument('--learning_rate_ave', type=float, default=1e-3)
parser.add_argument('--learning_rate_rs', type=float, default=1e-4)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--model_type', type=str, default='bert')
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--output_mode', type=str, default='classification')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--model_version', type=str, default='bert-base-uncased')
parser.add_argument('--MAX_EPOCHS', type=int, default=1000)
parser.add_argument('--num_train_epochs', type=int, default=4)
parser.add_argument('--per_checkpoint', type=int, default=10)
parser.add_argument('--attention_heads', type=int, default=8)
parser.add_argument('--do_train', type=bool, default=True)
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--device', type=str, default=device)
parser.add_argument('--n_gpu', type=int, default=torch.cuda.device_count())
parser.add_argument('--log_file', type=str, default='./log/log.txt')
FLAGS = parser.parse_args()

if FLAGS.dataset == 'IMDB':
    FLAGS.classes = [str(i) for i in range(0, 10)]
else:
    FLAGS.classes = [str(i) for i in range(0, 5)]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(train_set, inference):
    """Training the model"""
    running_loss = 0.0
    model.train() if not inference else model.eval()
    labels = []
    preds = []

    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=FLAGS.batch_size)

    total_steps = len(train_set) * FLAGS.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_group_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': FLAGS.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    optimizer_bert = AdamW(optimizer_group_parameters,
                           lr=FLAGS.learning_rate_bert,
                           eps=FLAGS.adam_epsilon,
                           )
    # design learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer_bert,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    for idx, batch in enumerate(train_set):
        inputs = {'input_ids': batch[0].unsqueeze(0),
                  'token_type_ids': batch[1].unsqueeze(0),
                  'attention_mask': batch[2].unsqueeze(0),
                  'labels': batch[3].unsqueeze(0)}
        outputs = model(**inputs)
        # print(outputs)
        loss = getVariable(outputs[0])
        logits = outputs[1]

        loss.backward()
        running_loss += loss.item()

        optimizer_bert.zero_grad()

        # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer_bert.step()
        scheduler.step()

        label = int(batch[3].data.item())
        # print(label)
        pred = torch.argmax(logits).data.item()
        # print(pred)

        labels.append(label)
        preds.append(pred)

    # print(labels)
    # print(preds)

    return running_loss / len(train_dataloader), labels, preds


def evaluate(dev_set):
    """Evaluate the model"""
    model.eval()
    loss = 0.0
    step = 0
    # dev_sampler = SequentialSampler(dev_set)
    # dev_dataloader = DataLoader(dev_set, sampler=dev_sampler, batch_size=FLAGS.eval_batch_size)
    with torch.no_grad():
        out_loss, true, pred = train(dev_set, inference=True)
    loss += out_loss
    step += 1

    accuracy = metric.accuracy_score(true, pred)
    precision = metric.precision_score(true, pred, average='weighted')
    recall = metric.recall_score(true, pred, average='weighted')
    f1_macro = metric.f1_score(true, pred, average='macro')
    f1_micro = metric.f1_score(true, pred, average='micro')
    MSE = metric.mean_squared_error(true, pred)
    RMSE = metric.mean_squared_error(true, pred, squared=False)
    c_m = metric.confusion_matrix(true, pred)

    loss /= step

    return loss, accuracy, precision, recall, f1_macro, f1_micro, MSE, RMSE, c_m


if __name__ == '__main__':
    set_seed(FLAGS)
    config = BertConfig.from_pretrained(
        FLAGS.model_version,
        num_labels=len(FLAGS.classes),
    )
    config.output_attention = False
    config.output_hidden_states = False
    config.attention_heads = FLAGS.attention_heads
    tokenizer = BertTokenizer.from_pretrained(FLAGS.model_version, do_lower_case=True)

    myProcessor = Datapreprocessor(FLAGS, tokenizer, config, FLAGS.data_path+FLAGS.dataset)

    # design datasets
    train_set = myProcessor.train_set
    dev_set = myProcessor.dev_set
    test_set = myProcessor.test_set
    print('train data: %s, val data: %s, test data: %s'
          % (len(train_set), len(dev_set), len(test_set)))

    data_examples = [train_set, dev_set, test_set]

    model = documentLevelBert(FLAGS, config)
    criterion = CrossEntropyLoss()

    loss_step = 0.0
    logger = Logger(FLAGS.log_file).get_log()

    start_time = time.time()
    for step in range(FLAGS.MAX_EPOCHS):
        if step % FLAGS.per_checkpoint == 0:
            time_elapsed = format_time(time.time() - start_time)
            start_time = time.time()
            print('================== CHECK ==================')
            print('Time of iter training: {}'.format(time_elapsed))
            print('On iter step %s: \t global step: %d \t learning rate: %.6f \t Loss-step: %s'
                  % (step / FLAGS.per_checkpoint, step, FLAGS.learning_rate_bert, loss_step))

            # Log file
            logger.info('%s Of %s' % (step / FLAGS.per_checkpoint, step))
            logger.info('BERT Learning Rate: %.4f' % FLAGS.learning_rate_bert)
            logger.info('Loss Step: %s' % loss_step)

            for idx, dataset in enumerate(data_examples):
                loss, acc, pre, rec, f1_macro, f1_micro, MSE, RMSE, c_m = evaluate(dataset)
                print('In %s: \n' % dataset)
                print('Loss = %s' % loss)
                print('Accuracy = %s' % acc)
                print('Precision = %s' % pre)
                print('Recall = %s' % rec)
                print('F1_macro = %s' % f1_macro)
                print('F1_micro = %s' % f1_micro)
                print('MSE = %s' % MSE)
                print('RMSE = %s' % RMSE)
                print('Confusion Matrix =')
                print(c_m)
                print('\n')

                logger.info('~ ~ ~ ~ ~ %s ~ ~ ~ ~ ~' % dataset)
                line = 'Accuracy: ' + str(acc) + '\n' \
                       + 'Precision: ' + str(pre) + '\n' \
                       + 'Recall: ' + str(rec) + '\n' \
                       + 'F1_macro: ' + str(f1_macro) + '\n' \
                       + 'F1_micro: ' + str(f1_micro) + '\n' \
                       + 'MSE: ' + str(MSE) + '\n' \
                       + 'RMSE: ' + str(RMSE) + '\n' \
                       + 'Confusion_matrix: ' + str(c_m) + '\n\n\n'
                logger.info(line)

            loss_step = 0.0

        loss_step += train(train_set, inference=False)[0] / FLAGS.per_checkpoint
