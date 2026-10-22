import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.sys_mat.analytic_sino as asino


def imshow_proj(proj_l, proj_v, fig_title='', idx=0):

    if proj_l.ndim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
        fig.suptitle(fig_title+ ' Projection', fontsize=14, fontweight='bold')
        ax1.imshow(proj_l.T, cmap='gray', aspect='equal', origin='lower')
        ax1.set_title('Line Intersection')
        ax1.set_xlabel('u')
        ax1.set_ylabel('v')
        
        ax2.imshow(proj_v.T, cmap='gray', aspect='equal', origin='lower')
        ax2.set_title('Volume Intersection')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    else:
        na, nu, nv = proj_l.shape
        
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
        fig.suptitle(fig_title + ' Projection', fontsize=14, fontweight='bold')
        ax1.imshow(proj_l[idx,:,:].T, cmap='gray', aspect='equal', origin='lower')
        ax1.set_title('Line Intersection')
        ax1.set_xlabel('u')
        ax1.set_ylabel('v')
        
        ax2.imshow(proj_v[idx,:,:].T, cmap='gray', aspect='equal', origin='lower')
        ax2.set_title('Volume Intersection')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')        
        fig.suptitle(fig_title + ' Sinogram', fontsize=14, fontweight='bold')
        ax1.imshow(proj_l[:,:,int(nv/2)], cmap='gray', aspect='equal', origin='lower')
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('Angles')
        
        ax2.imshow(proj_v[:,:,int(nv/2)], cmap='gray', aspect='equal', origin='lower')
        ax2.set_xlabel('Bins')
        ax2.set_ylabel('Angles')
        
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        """
    
    
    


def plot_proj(proj_l, proj_v, fig_title='', idx=0):

    if proj_l.ndim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
        fig.suptitle(fig_title+ ' Projection', fontsize=14, fontweight='bold')
        ax1.imshow(proj_l.T, cmap='gray', aspect='equal', origin='lower')
        ax1.set_title('Line Intersection')
        ax1.set_xlabel('u')
        ax1.set_ylabel('v')
        
        ax2.imshow(proj_v.T, cmap='gray', aspect='equal', origin='lower')
        ax2.set_title('Volume Intersection')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')
        
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()




    
#Geom
na = 4
angles = np.linspace(0,2*np.pi,na,endpoint=False)
DSO = 1000.0
DSD = 1500.0

#Detector
nu = 32
nv = 32

u = vir.censpace(nu)
v = vir.censpace(nv)
u_bnd = vir.boundspace(nu)
v_bnd = vir.boundspace(nv)


#Sphere
x,y,z = (0.5,0.0,0.001)
r = 3.9
harho = 1.0
s = (x,y,z,r)

#Cube
cube_center = (0.0,0.0,0.0)
cube_size = (100.0,100.0,100.0)
cube_planes = asino.cube_planes(cube_center, cube_size)



src0 = np.array([DSO,0.0,0.0])
det0 = np.array([DSO - DSD,0.0,0.0])
u_hat0 = np.array([0.0,1.0,0.0])
v_hat0 = np.array([0.0,0.0,1.0])
du = u[1] - u[0]
dv = v[1] - v[0]

dets0 = asino.detector_grid(det0, u, v, u_hat=u_hat0, v_hat=v_hat0)





test1=asino.analytic_sino_sphere(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0.0,0.0,0.0),r)

    
    
test2=asino.analytic_sino_sphere(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0.00001,0.00001,0.00001),r)

print("Centered sphere hemisphere volume", test1)
print("Centered sphere total volume:", test1.sum())
print("Centered sphere hemisphere volume", test2)
print("Shifted sphere total volume:", test2.sum())
print("Analytic volume:",r**3 *np.pi*4/3)
    

    
#test2=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
#                                (0.01,0.01,0.01),r)


print(test1.sum())
print(test2.sum())
print(r**3 *np.pi*4/3)
    
    
    
    


"""
L = 1000.0

planes = np.array([
    [ 1, 0, 0, L],
    [-1, 0, 0, L],
    [ 0, 1, 0, L],
    [ 0,-1, 0, L],
    [ 0, 0, 1, L],
    [ 0, 0,-1, L],
])

vol = asino.sphere_clip_volume(
    planes,
    center=np.array([0,0,0]),
    radius=50,
    dirs=dirs.t,
    weights=weights
)


"""
    
    
"""
###Sphere General###
projl_s_gen = asino.analytic_sino_sphere_gen(src0, dets0, s, rho=1.0)
projv_s_gen = asino.analytic_sino_sphere_gen_vol(src0, dets0, \
                                               u_hat0, v_hat0, du, dv, s).squeeze()
imshow_proj(projl_s_gen, projv_s_gen, fig_title='Sphere General')
"""







"""

###Sphere Cone###
polyl_s_cone = asino.analytic_sino_sphere_cone0(s, angles, u, v, DSO, DSD)
polyv_s_cone = asino.analytic_sino_sphere_cone_vol(s, angles, u, v, DSO, DSD)
imshow_proj(polyl_s_cone, polyv_s_cone, fig_title='Sphere Cone')

###Cube General###
polyl_p_gen = asino.analytic_sino_polyhedron_gen(src0, det0, u, v, u_hat0, v_hat0, cube_planes)
polyv_p_gen = asino.analytic_sino_polyhedron_gen_vol(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,cube_planes)
imshow_proj(polyl_p_gen, polyv_p_gen, fig_title='Cube General')


###Cube Cone###
polyl_p_cone = asino.analytic_sino_polyhedron_cone(cube_planes, angles, u, v, DSO, DSD)
polyv_p_cone = asino.analytic_sino_polyhedron_cone_vol(cube_planes, angles, u_bnd, v_bnd, DSO, DSD)
imshow_proj(polyl_p_cone, polyv_p_cone, fig_title='Cube Cone')

"""



#sino_par = asino.analytic_circle_sino_par_2d((x,y,r), angles,u)
#sino_fan = asino.analytic_circle_sino_fan_2d((x,y,r), angles, u, DSO, DSD)














"""
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(sino_par[0,:], label='Parallel Beam')
plt.plot(sino_fan[0,:], label='Fan Beam')
plt.plot(sino_cone[0,:,int(nv/2)], label='Cone Beam')
plt.legend()

plt.subplot(1,3,2)
plt.plot(sino_par[int(na/4),:], label='Parallel Beam')
plt.plot(sino_fan[int(na/4),:], label='Fan Beam')
plt.plot(sino_cone[int(na/4),:,int(nv/2)], label='Cone Beam')
plt.legend()

plt.subplot(1,3,3)
plt.plot(sino_par[int(na/8),:], label='Parallel Beam')
plt.plot(sino_fan[int(na/8),:], label='Fan Beam')
plt.plot(sino_cone[int(na/8),:,int(nv/2)], label='Cone Beam')
plt.legend()
plt.show
"""



"""
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(sino_par, cmap='gray', aspect='auto', origin='lower')
plt.title("Parallel Beam")
plt.subplot(1,3,2)
plt.imshow(sino_fan, cmap='gray', aspect='auto', origin='lower')
plt.title("Fan Beam")
plt.subplot(1,3,3)
plt.imshow(sino_cone[:,:,int(nv/2)], cmap='gray', aspect='auto', origin='lower')
plt.title("Cone Beam")
plt.show


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(sino_par[0,:], label='Parallel Beam')
plt.plot(sino_fan[0,:], label='Fan Beam')
plt.plot(sino_cone[0,:,int(nv/2)], label='Cone Beam')
plt.legend()

plt.subplot(1,3,2)
plt.plot(sino_par[int(na/4),:], label='Parallel Beam')
plt.plot(sino_fan[int(na/4),:], label='Fan Beam')
plt.plot(sino_cone[int(na/4),:,int(nv/2)], label='Cone Beam')
plt.legend()

plt.subplot(1,3,3)
plt.plot(sino_par[int(na/8),:], label='Parallel Beam')
plt.plot(sino_fan[int(na/8),:], label='Fan Beam')
plt.plot(sino_cone[int(na/8),:,int(nv/2)], label='Cone Beam')
plt.legend()
plt.show
"""

