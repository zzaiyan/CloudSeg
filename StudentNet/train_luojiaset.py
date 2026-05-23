import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from luojiaset_runners import BaseLuojiaTrainScript  # noqa: E402


class StudentLuojiaTrainScript(BaseLuojiaTrainScript):
    dataloader_module_name = "dataloader_luojiaset"
    model_module_name = "model_SS_net"
    generic_train_module_name = "generic_train"
    save_model_dir = "../checkpoints/StudentNet_Luojia"
    teacher_pretrained_model = "../checkpoints/TeacherNet_Luojia/best_semantic_net.pth"
    is_load_cloudmask = True


if __name__ == "__main__":
    StudentLuojiaTrainScript().run()
