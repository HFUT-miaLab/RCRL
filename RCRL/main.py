import copy
import os
import datetime
import argparse
import shutil
import sys
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

# local library
from Dataset_TCGA_EGFR import TCGA_EGFR_feat_label_Dataset, get_train_valid_names, get_test_names
import metric
from lookahead import Lookahead
from model import RCRL
from radam import RAdam
from utils import Logger


class BestModelSaver:
    def __init__(self, max_epoch, ratio=0.3):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_acc_epoch = 0
        self.best_valid_auc_epoch = 0

        self.begin_epoch = int(max_epoch * ratio)

    def update(self, valid_acc, valid_auc, current_epoch):
        if current_epoch < self.begin_epoch:
            return

        if valid_acc >= self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_acc_epoch = current_epoch
        if valid_auc >= self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_auc_epoch = current_epoch


def _macro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    if multi_class:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)
        return roc_auc_score(lbl_true, lbl_pred, average="macro", multi_class='ovo')
    else:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)[:, 1]

        return roc_auc_score(lbl_true, lbl_pred, average="macro")


def _micro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    if multi_class:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)
        return roc_auc_score(lbl_true, lbl_pred, average="micro", multi_class='ovr')
    else:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)[:, 1]
        return roc_auc_score(lbl_true, lbl_pred, average="micro")


def micro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    lbl_true = np.concatenate(lbl_true_list, axis=0)
    lbl_pred = np.concatenate(lbl_pred_list, axis=0)

    if multi_class:
        aucs = []
        binary_labels = label_binarize(lbl_true, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in lbl_true:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], lbl_pred[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        return np.nanmean(np.array(aucs))
    else:
        return roc_auc_score(lbl_true, lbl_pred)


def eval(args, test_loader, model, eval_metric='ACC'):
    model.eval()

    correct = 0
    total = 0

    lbl_true_list = []
    lbl_pred_list = []

    y_result = []
    pred_result = []
    pred_probs = []
    with torch.no_grad():
        for step, (x,text_feature,text_tokens_feature,sequence_feature, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                text_feature = text_feature.cuda()
                text_tokens_feature = text_tokens_feature.cuda()
                sequence_feature = sequence_feature.cuda()
                target = target.cuda()

            results_dict = model(x,text_feature,text_tokens_feature,sequence_feature,target)
            logits = results_dict['logits']

            pred_prob = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            correct = correct + int((prediction == target).sum().cpu())

            y_result = y_result + target.detach().cpu().tolist()
            pred_result = pred_result + prediction.tolist()
            pred_probs = pred_probs + pred_prob.tolist()

            lbl_true_list.append(target.detach().cpu().numpy())
            lbl_pred_list.append(pred_prob.detach().cpu().numpy())

            total += len(target)

    acc = correct / total

    if args.nclass == 2:
        class_names = ['Wild', 'Mutant']    #TCGA-EGFR
    elif args.nclass == 4:
        class_names = ['19del','L858R', 'Wild', 'Other']    #USTC-EGFR
    else:
        raise NotImplementedError

    metric.draw_confusion_matrix(y_result, pred_result, class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path, 'Confusion_Matrix_' + eval_metric + '.jpg'))

    macro_auc_score = _macro_auc(lbl_true_list, lbl_pred_list, multi_class=args.nclass > 2, n_classes=args.nclass)
    precision, recall, F1_score = precision_recall_fscore_support(y_result, pred_result, average='macro')[:-1]
    specificity = metric.compute_specificity(y_result, pred_result)
    sensitivity = metric.compute_sensitivity(y_result, pred_result)

    if args.nclass == 2:
        metric.draw_binary_roc_curve(y_result, np.array(pred_probs)[:, 1],
                                     save_path=os.path.join(args.fold_save_path, 'ROC_Curve_' + eval_metric + '.jpg'))
    else:
        metric.draw_muti_roc_curve(y_result, np.array(pred_probs), class_names=class_names,
                                     save_path=os.path.join(args.fold_save_path, 'ROC_Curve_' + eval_metric + '.jpg'))

    return acc, macro_auc_score, precision, recall ,F1_score, specificity, sensitivity


def valid(args, valid_loader, model):
    model.eval()

    correct = 0
    total = 0
    gts = []
    preds = []

    with torch.no_grad():
        for step, (x,text_feature,text_tokens_feature,sequence_feature, target) in enumerate(valid_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                text_feature = text_feature.cuda()
                text_tokens_feature = text_tokens_feature.cuda()
                sequence_feature = sequence_feature.cuda()
                target = target.cuda()

            results_dict = model(x,text_feature,text_tokens_feature,sequence_feature,target)
            logits = results_dict['logits']

            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            correct = correct + int((pred == target).sum().cpu())

            gts.append(target.detach().cpu().numpy())
            preds.append(probs.detach().cpu().numpy())

            total += len(target)

    acc = correct / total
    auc_score = _macro_auc(gts, preds, multi_class=args.nclass > 2, n_classes=args.nclass)

    return acc, auc_score

def compute_contrastive_loss( logits, list_len=5):
    contra_loss = contrastive_loss(logits, list_len)
    return contra_loss

def contrastive_loss(logits: torch.Tensor, list_len=5) -> torch.Tensor:
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits.unsqueeze(0), torch.tensor([list_len - 1], device=logits.device))

def train(args, model, train_loader, valid_loader):
    model.train()

    optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    optimizer = Lookahead(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)

    loss_fn = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn.cuda()

    args.current_epoch = 0
    best_model_saver = BestModelSaver(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        args.current_lr = lr

        train_loss = 0
        correct = 0
        total = 0
        total_step = 0

        for step, (x, text_feature,text_tokens_feature,sequence_feature, target) in enumerate(train_loader):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                x = x.cuda()
                text_feature = text_feature.cuda()
                text_tokens_feature = text_tokens_feature.cuda()
                sequence_feature = sequence_feature.cuda()
                target = target.cuda()

            results_dict = model(x,text_feature,text_tokens_feature,sequence_feature,target)   # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            target.unsqueeze(dim=0)
            correct = correct + int((pred == target).sum().cpu())
            total += 1
            L_pred = loss_fn(probs, target)

            image_loss = compute_contrastive_loss(results_dict['fusion_probs'], results_dict['length1'])
            sequence_loss = compute_contrastive_loss(results_dict['sequence_probs'], results_dict['length1'])
            L_mol = (image_loss + sequence_loss) / 2.0

            original_loss = compute_contrastive_loss(results_dict['original_probs'], results_dict['length2'])
            reconstruct_loss = compute_contrastive_loss(results_dict['reconstruct_probs'], results_dict['length2'])
            L_hist = (original_loss + reconstruct_loss) / 2.0

            total_loss = L_pred +  L_mol +  L_hist

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()

            total_step += 1
            if step % 50 == 0:
                print("\tEpoch: [{}/{}] || epochiter: [{}/{}] || Train Loss: {:.6f} || LR: {:.6f}"
                      .format(args.current_epoch + 1, args.epochs, step + 1, len(train_loader),
                              train_loss / total_step, args.current_lr))

        train_acc = correct / total

        valid_acc, valid_auc = valid(args, valid_loader, model)
        best_model_saver.update(valid_acc, valid_auc, args.current_epoch)
        print('\tValidation-Epoch: {} || train_acc: {:.6f} || train_avg_loss: {:.6f} ||'
              ' valid_acc: {:.6f} || valid_auc: {:.6f}'
              .format(args.current_epoch + 1, train_acc, train_loss / total_step, valid_acc, valid_auc))

        current_model_weight = copy.deepcopy(model.state_dict())
        torch.save(current_model_weight,
                   os.path.join(args.fold_save_path, 'epoch' + str(args.current_epoch) + '.pth'))

        args.current_epoch += 1

    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc.pth'))
    return best_model_saver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RCRL on TCGA_EGFR training script')
    parser.add_argument('--features_root', type=str,
                        default=r'...',help='patches feature save dir')
    parser.add_argument('--report_features_root', type=str,
                        default=r'...',help='histopathological report feature save dir')
    parser.add_argument('--report_tokens_features_root', type=str,
                        default=r'...',help='word tokens of histopathological report save dir')
    parser.add_argument('--sequence_features_root', type=str,
                        default=r'...',help='gene sequencing report feature save dir')
    parser.add_argument('--patch_dim', type=int, default=512, help='patches feature dimension')
    parser.add_argument('--text_dim', type=int, default=512, help='text feature dimension')
    parser.add_argument('--text_tokens_dim', type=int, default=768, help='word toknes feature dimension')
    parser.add_argument('--save_path', type=str, default=r".\weights_RCRL" ,help='model weight save dir')
    parser.add_argument('--logger_path', type=str, default=r'.\logger_RCRL' ,help='model training record save dir')
    parser.add_argument('--train_valid_csv', type=str, default=r"..." ,help='five-fold train_valid dataset')
    parser.add_argument('--test_csv', type=str, default=r"..." ,help='test dataset')
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=4, help='The length of contrastive learning queue(list)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--reg', type=float, default=1e-6)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    args.weights_save_path = os.path.join(args.save_path,datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)

    # 运行日志
    os.makedirs(args.logger_path, exist_ok=True)
    sys.stdout = Logger(filename=os.path.join(args.logger_path, datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    test_acc_5fold_acc_model = []
    test_macro_auc_5fold_acc_model = []
    test_precision_5fold_acc_model = []
    test_recall_5fold_acc_model = []
    test_F1_score_5fold_acc_model = []
    test_specificity_5fold_acc_model = []
    test_sensitivity_5fold_acc_model = []

    test_acc_5fold_auc_model = []
    test_macro_auc_5fold_auc_model = []
    test_precision_5fold_auc_model = []
    test_recall_5fold_auc_model = []
    test_F1_score_5fold_auc_model = []
    test_specificity_5fold_auc_model = []
    test_sensitivity_5fold_auc_model = []

    for fold in range(5):
        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)

        print('Training Folder: {}.\n\tData Loading...'.format(fold))
        train_names, train_labels, valid_names, valid_labels, train_weights = get_train_valid_names(args.train_valid_csv,
                                                                                        fold=fold,
                                                                                        nclass=args.nclass)
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))
        train_dataset = TCGA_EGFR_feat_label_Dataset(args.features_root,args.report_features_root,args.report_tokens_features_root,args.sequence_features_root, train_names, train_labels)
        valid_dataset = TCGA_EGFR_feat_label_Dataset(args.features_root,args.report_features_root,args.report_tokens_features_root,args.sequence_features_root, valid_names, valid_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)


        model = RCRL(n_classes=args.nclass, max_length=args.max_length,
                     patch_dim=args.patch_dim, text_dim=args.text_dim, text_tokens_dim=args.text_tokens_dim)

        if torch.cuda.is_available():
            model = model.cuda()
        best_model_saver = train(args, model, train_loader, valid_loader)
        print('\t(Valid)Best ACC: {:.6f} at {} epoch || Best AUC: {:.6f} at {} epoch'
              .format(best_model_saver.best_valid_acc, best_model_saver.best_valid_acc_epoch+ 1,
                      best_model_saver.best_valid_auc, best_model_saver.best_valid_auc_epoch+ 1))

        test_names, test_labels = get_test_names(args.test_csv, nclass=args.nclass)
        test_dataset = TCGA_EGFR_feat_label_Dataset(args.features_root,args.report_features_root,args.report_tokens_features_root,args.sequence_features_root, test_names, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        best_acc_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_acc.pth'))
        model.load_state_dict(best_acc_model_weight)
        test_acc, test_macro_auc, test_precision, test_recall, test_F1_score, test_specificity, test_sensitivity = eval(args, test_loader, model,
                                                                                    eval_metric='ACC')
        test_acc_5fold_acc_model.append(test_acc)
        test_macro_auc_5fold_acc_model.append(test_macro_auc)
        test_precision_5fold_acc_model.append(test_precision)
        test_recall_5fold_acc_model.append(test_recall)
        test_F1_score_5fold_acc_model.append(test_F1_score)
        test_specificity_5fold_acc_model.append(test_specificity)
        test_sensitivity_5fold_acc_model.append(test_sensitivity)

        print('\t(Test)Best ACC model || ACC: {:.6f} || Macro_AUC: {:.6f} || precision: {:.6f} || recall: {:.6f} || F1_score: {:.6f} || specificity: {:.6f} || sensitivity: {:.6f}\n'
            .format(test_acc, test_macro_auc, test_precision, test_recall, test_F1_score, test_specificity, test_sensitivity))

        best_auc_model_weight = torch.load(os.path.join(args.fold_save_path, 'best_auc.pth'))
        model.load_state_dict(best_auc_model_weight)
        test_acc, test_macro_auc, test_precision, test_recall, test_F1_score, test_specificity, test_sensitivity = eval(args, test_loader, model,eval_metric='AUC')
        test_acc_5fold_auc_model.append(test_acc)
        test_macro_auc_5fold_auc_model.append(test_macro_auc)
        test_precision_5fold_auc_model.append(test_precision)
        test_recall_5fold_auc_model.append(test_recall)
        test_F1_score_5fold_auc_model.append(test_F1_score)
        test_specificity_5fold_auc_model.append(test_specificity)
        test_sensitivity_5fold_auc_model.append(test_sensitivity)

        print('\t(Test)Best AUC model || ACC: {:.6f} || Macro_AUC: {:.6f} || precision: {:.6f} || recall: {:.6f} || F1_score: {:.6f} || specificity: {:.6f} || sensitivity: {:.6f}\n'
            .format(test_acc, test_macro_auc, test_precision, test_recall, test_F1_score, test_specificity, test_sensitivity ))

    print("Five-Fold-Validation:")
    print("\tBest_ACC_Model: ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2}, precision: {:.2f}±{:.2}, recall: {:.2f}±{:.2}, F1_score: {:.2f}±{:.2}, specificity: {:.2f}±{:.2}, sensitivity: {:.2f}±{:.2}"
        .format(np.mean(test_acc_5fold_acc_model) * 100, np.std(test_acc_5fold_acc_model) * 100,
                np.mean(test_macro_auc_5fold_acc_model) * 100, np.std(test_macro_auc_5fold_acc_model) * 100,
                np.mean(test_precision_5fold_acc_model) * 100, np.std(test_precision_5fold_acc_model) * 100,
                np.mean(test_recall_5fold_acc_model) * 100, np.std(test_recall_5fold_acc_model) * 100,
                np.mean(test_F1_score_5fold_acc_model) * 100, np.std(test_F1_score_5fold_acc_model) * 100,
                np.mean(test_specificity_5fold_acc_model) * 100, np.std(test_specificity_5fold_acc_model) * 100,
                np.mean(test_sensitivity_5fold_acc_model) * 100, np.std(test_sensitivity_5fold_acc_model) * 100))
    print("\tBest_AUC_Model: ACC: {:.2f}±{:.2f}, Macro_AUC: {:.2f}±{:.2}, precision: {:.2f}±{:.2}, recall: {:.2f}±{:.2}, F1_score: {:.2f}±{:.2}, specificity: {:.2f}±{:.2}, sensitivity: {:.2f}±{:.2}"
        .format(np.mean(test_acc_5fold_auc_model) * 100, np.std(test_acc_5fold_auc_model) * 100,
                np.mean(test_macro_auc_5fold_auc_model) * 100, np.std(test_macro_auc_5fold_auc_model) * 100,
                np.mean(test_precision_5fold_auc_model) * 100, np.std(test_precision_5fold_auc_model) * 100,
                np.mean(test_recall_5fold_auc_model) * 100, np.std(test_recall_5fold_auc_model) * 100,
                np.mean(test_F1_score_5fold_auc_model) * 100, np.std(test_F1_score_5fold_auc_model) * 100,
                np.mean(test_specificity_5fold_auc_model) * 100, np.std(test_specificity_5fold_auc_model) * 100,
                np.mean(test_sensitivity_5fold_auc_model) * 100, np.std(test_sensitivity_5fold_auc_model) * 100))
