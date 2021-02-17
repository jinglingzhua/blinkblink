from forward import Forward
from MyUtils.my_utils import *

class Perf:
    def __init__(self, predictor, imgsrc, perf_dir, err_save=False):
        os.makedirs(perf_dir, exist_ok=True)
        self.predictor, self.imgsrc, self.perf_dir = predictor, imgsrc, perf_dir
        self.err_save = err_save
        
    def proc(self):
        accu = 0
        images = get_spec_files(self.imgsrc, ext=IMG_EXT, oswalk=True)
        for i, img in enumerate(images):
            print('{}/{}'.format(i, len(images)))
            I = cv2.imread(img, 1)
            _, fname = os.path.split(img)
            gt = int(fname.split('_')[4])
            score = self.predictor(I)
            if (score >= 0.5 and gt == 1) or (gt < 0.5 and gt == 0):
                accu += 1
                
        with open(opj(self.perf_dir, 'result.csv'), 'w') as f:
            f.write('{}\n'.format(accu/len(images)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument("perf_dir")
    parser.add_argument("--model_path")
    parser.add_argument("--model_cfg")
    args = parser.parse_args()

    predictor = Forward(args.model_path, args.model_cfg)
    perf = Perf(predictor, args.src, args.perf_dir, err_save=False)
    perf.proc()








