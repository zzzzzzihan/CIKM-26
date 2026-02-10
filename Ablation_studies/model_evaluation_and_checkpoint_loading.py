# 测试评估代码（你需要运行）
import torch
from pytorch_lightning import Trainer

# 加载最佳模型
model = QTFT.load_from_checkpoint("outputs/QTFT_after_vs/version_0/checkpoints/epoch=49-step=350.ckpt")
model.eval()

# 创建测试加载器
test_loader = DataLoader(test_dataset, batch_size=32)

# 评估
trainer = Trainer()
results = trainer.test(model, test_loader)

# 保存结果
import json
with open("test_results.json", "w") as f:
    json.dump(results[0], f, indent=2)