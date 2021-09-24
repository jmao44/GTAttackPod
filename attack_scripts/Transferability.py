import sys
sys.path.append(".")
from attacks import *
from datasets import *
from models import *
import time


if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    model_1 = CIFAR10_resnet20(rel_path='./')
    model_2 = CIFAR10_densenet40(rel_path='./')
    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model_2,
                                                                                                       balanced=True,
                                                                                                       num_examples=100)

    deepfool = Attack_DeepFool(overshoot=10)
    time_start = time.time()
    X_test_adv = deepfool.attack(model_1, X_test, Y_test)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    # Evaluate the adversarial examples.
    print("\n---Statistics of DeepFool Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model_1.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)
