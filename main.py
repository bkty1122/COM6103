from attack.backdoor import BackdoorAttack
from attack.poisoned import PoisonedAttack
from defense.adversarial import AdversarialDefense
from defense.dataCleaning import DataCleaning
from defense.detectBackdoor import DetectBackdoor
from modelDataImport import load_data
from modelTrain import train_model
from modelEval import evaluate_model

def main():
    # 1. 加载数据
    print("Loading data...")
    train_loader, test_loader = load_data()

    # 2. 数据投毒攻击
    print("Performing data poisoning attack...")
    poisoned_attack = PoisonedAttack(train_loader, poison_ratio=0.1, target_label=0)
    poisoned_train_loader = poisoned_attack.apply_attack()

    # 3. 后门攻击
    print("Performing backdoor attack...")
    backdoor_attack = BackdoorAttack(train_loader, trigger_ratio=0.1, target_label=0)
    backdoor_train_loader = backdoor_attack.apply_attack()

    # 4. 数据清洗
    print("Cleaning data...")
    data_cleaning = DataCleaning(train_loader)
    clean_train_loader = data_cleaning.clean_data()

    # 5. 对抗训练
    print("Performing adversarial training...")
    adversarial_defense = AdversarialDefense(clean_train_loader)
    model = adversarial_defense.train_model()

    # 6. 后门检测
    print("Detecting backdoor...")
    backdoor_detector = DetectBackdoor(model, test_loader, backdoor_train_loader)
    backdoor_detector.detect()

    # 7. 训练模型
    print("Training model...")
    trained_model = train_model(clean_train_loader)

    # 8. 评估模型
    print("Evaluating model...")
    normal_accuracy = evaluate_model(trained_model, test_loader)
    backdoor_accuracy = evaluate_model(trained_model, backdoor_train_loader)
    print(f"Normal Accuracy: {normal_accuracy:.2f}%")
    print(f"Backdoor Accuracy: {backdoor_accuracy:.2f}%")

if __name__ == "__main__":
    main()