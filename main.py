import argparse
import yaml
from src.models import CNNTeacherModel,StudentModel,CrossWindowAttentionModel
from src.train import train_teacher,train_student_kd_fixed,train_baseline_model
from src.data_utils import download_data,load_Data,create_Dataloader
from src.eval import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate model")
    
    parser.add_argument('--config', type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument('--train', action='store_true', help="Flag to train model")
    parser.add_argument('--evaluate', action='store_true', help="Flag to evaluate model")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training/evaluation")
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()

    config = load_config(args.config)
    
    records_url = "https://physionet.org/files/mimic3wdb/1.0/30/RECORDS"
    records_local = "RECORDS_30.txt"
    download_dir = "usable_records"
    
    download_data(records_url, records_local,download_dir)
    
    x,y = load_data(download_dir)
    
    train_loader,test_loader = create_dataloader(x,y, batch_size=args.batch_size)
    
    teacher_model = CNNTeacherModel()
    student_model = StudentModel()
    baseline_model = BaselineECGModel().to(device)

    if args.train:
        print(f"Training model for {args.epochs} epochs with batch size {args.batch_size}")
        train_teacher(teacher_model, train_loader,args.epochs)
        torch.save(model.state_dict(), "teacher_model_weights.pth")
        train_student_kd_fixed(student_model,teacher_model, train_loader, args.epochs)
        torch.save(train_student_kd_fixed.state_dict(), "student_model_weights.pth")
        train_baseline_model()
        torch.save(baseline_model.state_dict(), "baseline_model_weights.pth")
    elif args.evaluate:
        print("Evaluating model")
        evaluate_model(student_model, test_loader)
        evaluate_model(teacher_model, test_loader)
        evaluate_model(baseline_model, test_loader)
    else:
        print("Please specify --train or --evaluate")

if __name__ == "__main__":
    main()
