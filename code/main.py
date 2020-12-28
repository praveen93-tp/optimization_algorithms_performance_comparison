import argparse
from copy import deepcopy
import torch
import temp_config_files
import optimizers
from neural_networks_architecture import MLP, CNN, train
from torchsummary import summary
import xlwt
from xlwt import Workbook
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=30)
    parser.add_argument('-dataset', type=str, default='mnist')
    parser.add_argument('-num_train', type=int, default=50000)
    parser.add_argument('-num_val', type=int, default=2048)
    parser.add_argument('-lr_schedule', type=bool, default=True)
    parser.add_argument('-only_plot', type=bool, default=True)
    args = parser.parse_args()
    data = getattr(temp_config_files, 'load_'+args.dataset)(num_train=args.num_train,num_val=args.num_val)
    print(f'Loaded data partitions: ({len(data[0])}), ({len(data[1])})')
    optimization_algorithms_implemented = [
        'padam','amsgrad','adamax','adabound','adabound_w','sgd','sgd_momentum','sgd_nesterov','sgd_weight_decay','sgd_lrd','rmsprop','adam','adam_l2','adamW','adam_lrd',
        'Radam','RadamW','Radam_lrd','nadam','lookahead_sgd','lookahead_adam','gradnoise_adam','graddropout_adam','hessian_free'
    ]
    opt_losses, opt_val_losses, opt_labels = [], [], []
    wb = Workbook()
    def run_optimization_alogrithm(opt):
        print(f'\nTraining {opt} for {args.num_epochs} epochs...')
        model = CNN() if args.dataset == 'cifar' else MLP()
        _, kwargs = temp_config_files.split_optim_dict(temp_config_files.optim_dict[opt])
        optimizer = temp_config_files.task_to_optimizer(opt)(params=model.parameters(),**kwargs)
        optimizer = temp_config_files.wrap_optimizer(opt, optimizer)
        return train(model, data, optimizer, num_epochs=args.num_epochs, lr_schedule=True)

    for opt in optimization_algorithms_implemented:
        if args.only_plot:
            losses = temp_config_files.load_losses(dataset=args.dataset, filename=opt)
            val_losses = temp_config_files.load_losses(dataset=args.dataset, filename=opt+'_val')
            losses, val_losses, pp_train, pp_loss, tr_ac, vl_ac = run_optimization_alogrithm(opt)
            temp_config_files.save_losses(losses, dataset=args.dataset, filename=opt)
            plt.plot(pp_train, 'g', label='Training loss')
            plt.plot(pp_loss, 'b', label='validation loss')
            plt.title(opt + ': ' + 'Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('Plots/' + opt + '_' + 'loss.png')
            plt.clf()
            sheet1 = wb.add_sheet(opt)
            for i in range(len(pp_train)):
                sheet1.write(i + 1, 0, pp_train[i])
                sheet1.write(i + 1, 1, pp_loss[i])
                sheet1.write(i + 1, 2, tr_ac[i])
                sheet1.write(i + 1, 3, vl_ac[i])
            wb.save('Remember/' + opt + '.xls')
            temp_config_files.save_losses(val_losses, dataset=args.dataset, filename=opt+'_val')
        if losses is not None:
            opt_losses.append(losses)
            opt_val_losses.append(val_losses)
            opt_labels.append(temp_config_files.split_optim_dict(temp_config_files.optim_dict[opt])[0])
