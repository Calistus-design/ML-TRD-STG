# THIS is a wrapper file to call ModelTrainer from src/model_trainer.py
from src.model_trainer import ModelTrainer
if __name__ == "__main__":
    trainer = ModelTrainer(data_path='data/training_data_4.csv')
    trainer.train(target_col='target_3m_call')
    #trainer.run_tournament(target_col='target_3m_call')