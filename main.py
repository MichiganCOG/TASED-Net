import sys
import os
import numpy as np
import cv2
import torch
from model import TASED_v2
from scipy.ndimage.filters import gaussian_filter

def main():
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    # optional two command-line arguments
    path_indata = './example'
    path_output = './output'
    if len(sys.argv) > 1:
        path_indata = sys.argv[1]
        if len(sys.argv) > 2:
            path_output = sys.argv[2]
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    len_temporal = 32
    file_weight = './TASED_v2.pt'

    model = TASED_v2()

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    for dname in list_indata:
        print ('processing ' + dname)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname)) if os.path.isfile(os.path.join(path_indata, dname, f))]
        list_frames.sort()

        # process in a sliding window fashion
        if len(list_frames) >= 2*len_temporal-1:
            path_outdata = os.path.join(path_output, dname)
            if not os.path.isdir(path_outdata):
                os.makedirs(path_outdata)

            snippet = []
            for i in range(len(list_frames)):
                img = cv2.imread(os.path.join(path_indata, dname, list_frames[i]))
                img = cv2.resize(img, (384, 224))
                img = img[...,::-1]
                snippet.append(img)

                if i >= len_temporal-1:
                    clip = transform(snippet)

                    process(model, clip, path_outdata, i)

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        process(model, torch.flip(clip, [1]), path_outdata, i-len_temporal+1)

                    del snippet[0]

        else:
            print (' more frames are needed')


def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)


def process(model, clip, path_outdata, idx):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]

    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=7)
    cv2.imwrite(os.path.join(path_outdata, '%04d.png'%(idx+1)), (smap/np.max(smap)*255.).astype(np.uint8))


if __name__ == '__main__':
    main()

