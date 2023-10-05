# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from config.base import Config
from torch.utils.data import DataLoader
import numpy as np
import torch

class runExperiment():

    def __init__(self, model_configuration, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.exp_path = exp_path

    def run_test(self, train_data,val_data,test_data, logger,contamination,query_num):

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        network = self.model_config.network
        trainer_class = self.model_config.trainer
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        try:
            x_dim = self.model_config['x_dim']
        except:
            x_dim = train_data.dim_features
        try:
            batch_size = self.model_config['batch_size']
        except:
            batch_size = int(np.ceil(len(train_data)/4))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=False)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    drop_last=False)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     drop_last=False)


        #### BULE
 
         # if the simple_refine method is used, in the first step the model is trained on the whole training set
         # in the second step the model only trains on a reduced training set
        if self.model_config['train_method'] == 'simple_refine':
            
            model = model_class(network(),x_dim, config=self.model_config)
            optimizer = optim_class(model.parameters(),
                                    lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

            if sched_class is not None:
                scheduler = sched_class(optimizer)
            else:
                scheduler = None

            trainer = trainer_class(model, loss_function=loss_class(self.model_config['loss_temp']),
                            config=self.model_config)
            # train model with full training set
            _,_,_,_,_,_,_ = \
            trainer.train(train_loader=train_loader,
                      contamination=contamination,query_num = query_num,
                      optimizer=optimizer, scheduler=scheduler,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)
            

            trainer.kwargs['simple_refine_flag']=True

            score_all=trainer.detect_outliers(train_loader)

            # remove the most anomalous samples from the training set here it is the contamination
            _, idx_n = torch.topk(torch.Tensor(score_all), int(score_all.shape[0] * (1-contamination)), largest=False,sorted=False)
            
            # save idx as pickle file
            torch.save(idx_n, '/root/LatentOE-AD/DATA/idx_simple refine' + '/idx_n.pkl')
            print(idx_n)

            train_data.labels= train_data.labels[idx_n]
            train_data.samples= train_data.samples[idx_n]
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,drop_last=False)


        model = model_class(network(),x_dim, config=self.model_config)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None


        trainer = trainer_class(model, loss_function=loss_class(self.model_config['loss_temp']),
                         config=self.model_config)


        val_loss,val_auc,test_auc,test_ap,test_f1,test_score,target_all = \
            trainer.train(train_loader=train_loader,
                      contamination=contamination,query_num = query_num,
                      optimizer=optimizer, scheduler=scheduler,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)

        return val_auc, test_auc, test_ap,test_f1, test_score,target_all
