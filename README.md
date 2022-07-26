Official repository of the paper (AI for Content Creation (AICC) Workshop, CVPR 2022) ["Image2Gif: Generating Continuous Realistic Animations with Warping NODEs"](demo/paper.pdf), Warping Neural ODE.

If you like our work, please give us a star. If you use our work in your research projects,
please cite our paper as
```
@article{nazarovs2022image2gif,
  title={Image2Gif: Generating Continuous Realistic Animations with Warping NODEs},
  author={Nazarovs, Jurijs and Huang, Zhichun},
  journal={arXiv preprint arXiv:2205.04519},
  year={2022}
}
```

![](demo/images/smile_teaser.png "Demonstration of Warping Neural ODE" )
![](demo/images/smile_vf.png "Demonstration of Warping Neural ODE" )
![](demo/images/warpode.png "Demonstration of Warping Neural ODE" )


# Visualization of experiments
## Warping MNIST
To train the model on warping from 3 to 4, you can run the folloowing code:
```python
python3 main.py --n_epochs 5000 --data mnist_warp --experimentID mnist_warp --device 0  --batch_size 64  --n_epochs_start_viz 50000 --gen_loss_weight 1 --disc_loss_weight 1 --gan_type lsgan --ode_solver euler  --freq_gen_update 1 --decay_every 100 --lr_gen 0.0001 --lr_disc 0.0001 --ode_vf init_img_y --ode_norm none --plots_path plots/ --rec_loss_weight 0.01 --freq_rec_update 1 --last_warp --jac_loss_weight_forw 1 --jac_loss_weight_back 1 --rec_weight_method default --outgrid_loss_weight_forw 1 --outgrid_loss_weight_back 1 --crit_iter 10 --gplambda 10 --disc_optim adam --unet_add_input --normalize_method scale --ode_step_size 0.05 --channels 1
```
Once training is done, you can add to the previous command the following arguments:
```--load --test_only --plot_last_vf --plot_all ```
It will load the last model, and produce images of warping in the following directory: `plots/mnnist_warp/vf_seq/`. \\
Then you can use `make_gif.sh` scrip to generate gif as (for example): 
`./makegif.sh plots_gif/mnist.gif plots/mnist_warp/vf_seq/0_0/ 2 100`.

![](demo/images/mnist.gif)

## Smiling face 
Similar, to generate smiling faces you can run the following code:
```python
python3 main.py --n_epochs 5000 --data face_warp_smile --experimentID face_warp_smile --device 0  --batch_size 16  --n_epochs_start_viz 50000 --gen_loss_weight 1 --disc_loss_weight 1 --gan_type None --ode_solver euler  --freq_gen_update 1 --decay_every 100 --lr_gen 0.0001 --lr_disc 0.0001 --ode_vf init_img_y --ode_norm none --plots_path plots/ --rec_loss_weight 0.01 --freq_rec_update 1 --last_warp --jac_loss_weight_forw 1 --jac_loss_weight_back 1 --rec_weight_method default --outgrid_loss_weight_forw 1 --outgrid_loss_weight_back 1 --crit_iter 10 --gplambda 10 --disc_optim adam --unet_add_input --normalize_method scale --ode_step_size 0.05 --channels 4 
```
Once training is done, you can add to the previous command the following arguments:
```--load --test_only --plot_last_vf --plot_all ```
It will load the last model, and produce images of warping in the following directory: `plots/face_warp_smile/vf_seq/`. \\
Then you can use `make_gif.sh` scrip to generate gif as (for example): 
`./makegif.sh plots_gif/smile.gif plots/face_warp_smile/vf_seq/0_0/ 2 100`.

![](demo/images/smile_0.gif) ![](demo/images/smile_1.gif)
![](demo/images/smile_2.gif)
![](demo/images/smile_3.gif)
![](demo/images/smile_4.gif)
![](demo/images/smile_5.gif)
![](demo/images/smile_6.gif)
![](demo/images/smile_7.gif)
