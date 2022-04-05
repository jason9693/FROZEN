import glob
import os

if __name__ == "__main__":
    src_list = glob.glob('/nas/po.ai/BiFrost*.ckpt')
    os.makedirs('/project/BiFrost', exist_ok=True)
    for src in src_list:
        filename = src.split('/')[-1]
        dst = f'/project/BiFrost/{filename}'
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, f'/project/BiFrost/{filename}')
