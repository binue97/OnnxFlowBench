import torch.utils.data as data

from dataloader.chairs import FlyingChairs
from dataloader.things import FlyingThings
from dataloader.sintel import MpiSintel
from dataloader.kitti import KITTI
from dataloader.spring import Spring
from dataloader.hd1k import HD1K
from dataloader.tartanair import TartanAir


def fetch_dataloader(args):
    """ Create the data loader for the corresponding evaluation set """

    if args.dataset == 'chairs':
        train_dataset = FlyingChairs(split='training')
    
    elif args.dataset == 'things':
        clean_dataset = FlyingThings(dstype='frames_cleanpass')
        final_dataset = FlyingThings(dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    
    elif args.dataset == 'cplust':
        chairs_dataset = FlyingChairs(split='training')
        clean_dataset = FlyingThings(dstype='frames_cleanpass')
        final_dataset = FlyingThings(dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset + chairs_dataset

    elif args.dataset == 'sintel':
        sintel_clean = MpiSintel(split='training', dstype='clean')
        sintel_final = MpiSintel(split='training', dstype='final')
        train_dataset = sintel_clean + sintel_final

    elif args.dataset == 'kitti':
        train_dataset = KITTI(split='training')
    
    elif args.dataset == 'spring':
        train_dataset = Spring(split='train') + Spring(split='val')

    elif args.dataset == 'tartanair':
        train_dataset = TartanAir()

    elif args.dataset == 'TSKH':
        things = FlyingThings(dstype='frames_cleanpass') + FlyingThings(dstype='frames_finalpass')
        sintel_clean = MpiSintel(split='training', dstype='clean')
        sintel_final = MpiSintel(split='training', dstype='final')
        kitti = KITTI()
        hd1k = HD1K()
        train_dataset = 20 * sintel_clean + 20 * sintel_final + 80 * kitti + 30 * hd1k + things

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
        pin_memory=False, shuffle=False, num_workers=8, drop_last=False)

    print('Evaluating with %d image pairs' % len(train_dataset))
    return train_loader
