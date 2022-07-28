def get_datapath(args):
    #-----------------------------------------------------------------------
    # Data definition
    #-----------------------------------------------------------------------
    # Need to clean up this block later with path to files
    data_path = {'source': {}, 'target': {}}
    if 'face_warp' in args.data:
        attr = args.data.split("_")[-1]

        data_path['source']['data'] = 'datasets/%s/neg' % attr
        data_path['target']['data'] = 'datasets/%s/pos' % attr
        args.img_size = 64
        args.channels = 4  #RGB and alpha

    elif args.data == 'mnist_warp':
        data_path['source']['data'] = 'datasets/RotatingMnist/source_3/'
        data_path['target']['data'] = 'datasets/RotatingMnist/source_4'
        args.img_size = 32
        args.channels = 1

    return data_path
