import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from luojiaset_runners import BaseLuojiaTrainScript  # noqa: E402


class TeacherLuojiaTrainScript(BaseLuojiaTrainScript):
    dataloader_module_name = "dataloader_luojiaset"
    model_module_name = "model_SS_net"
    generic_train_module_name = "generic_train"
    save_model_dir = "../checkpoints/TeacherNet_Luojia"
    is_load_cloudmask = False


if __name__ == "__main__":
    TeacherLuojiaTrainScript().run()
