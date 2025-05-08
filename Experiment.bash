#!/bin/bash

# Comando para executar o script em background (libera o terminal)
# chmod +x Experiment.bash && nohup setsid ./Experiment.bash > /dev/null 2>&1 &

# Configuração de parâmetros
DATASET="iot23-mirai-7-1-processed.csv"
DATASET_PATH="SourceDatasets/Processed/${DATASET}"
EPOCHS=20
MODEL_CHKPT="scarf1_embdd_dim=45_lr=0.001_bs=2046_epochs=${EPOCHS}_tempr=0.5_V=onlyunsw_cr_rt=0.4_ach_cr_rt0.2_msk_rt0_ach_msk_rt0.pth"
RENAMED_CHKPT="iot23-chpt.pth"
SUPERVISED_MODEL="RF"
RESULTS_FOLDER="Results/IoT23"
RESULTS_FILE="${RESULTS_FOLDER}/results.txt"
EVALUTAION_EPOCHS_RESULTS_FILE="${RESULTS_FOLDER}/evaluation_metrics_by_epoch.txt"
EVALUTAION_SUPERVISED_EPOCHS_RESULTS_FILE="${RESULTS_FOLDER}/evaluation_supervised_metrics_by_epoch.txt"
TERMINAL_LOGS_FILE="${RESULTS_FOLDER}/terminal_logs.txt"

# Criando os arquivos e diretórios
rm -f "new_checkpoints/$MODEL_CHKPT" "checkpoints/$RENAMED_CHKPT"
rm -rf $RESULTS_FOLDER
mkdir -p $RESULTS_FOLDER
touch $RESULTS_FILE
touch $TERMINAL_LOGS_FILE

# Treinamento
echo "Starting training..." >> $TERMINAL_LOGS_FILE
python3 train.py --dataset_path "$DATASET_PATH" --batch_size "2046" --epochs "$EPOCHS" >> $TERMINAL_LOGS_FILE 2>&1
mv "new_checkpoints/${MODEL_CHKPT}" "checkpoints/${RENAMED_CHKPT}"
echo "#################################################################################" >> $TERMINAL_LOGS_FILE

# Avaliação
echo "Starting evaluation..." >> $TERMINAL_LOGS_FILE
python3 evaluation.py --test_dataset_name "$DATASET_PATH" --batch_size "2046" --model_chkpt "$RENAMED_CHKPT" --train_from_scratch False --results_file "$RESULTS_FILE" >> $TERMINAL_LOGS_FILE 2>&1
echo "#################################################################################" >> $TERMINAL_LOGS_FILE

# Avaliação supervisionada
echo "Starting supervised evaluation..." >> $TERMINAL_LOGS_FILE
python3 evaluation_supervised.py --supervised_training_dataset "$DATASET_PATH" --batch_size "2046" --model_chkpt "$RENAMED_CHKPT" --supervised_model "$SUPERVISED_MODEL" --results_file "$RESULTS_FILE" >> $TERMINAL_LOGS_FILE 2>&1
echo "#################################################################################" >> $TERMINAL_LOGS_FILE

echo "All processes completed." >> $TERMINAL_LOGS_FILE
