import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#plt.rcParams['animation.ffmpeg_path'] = '/data/scratch3/src/ffmpeg'#'/opt/local/bin/ffmpeg' #tell matplotlib where ffmpeg lives

def solvr(X_V_arr, t, gamma, delta, beta, alpha, w):
    """Define the right-hand side of equation dx/dt = v 
       and dv/dt = gamma*np.cos(w*t) - delta*v - beta*x**3) - alpha*x """
    # The X_V_arr holds [position,velocity], ie [x, dx/dt]
    dx_dt = X_V_arr[1] # velocity
    dv_dt = gamma*np.cos(w*t) - delta*X_V_arr[1] - beta*(X_V_arr[0]**3) - alpha*X_V_arr[0]
    return[ dx_dt, dv_dt ] 

#Define a callback function
def _update_plot(i, Poincare_sec): 
    # i is the frame number, line is the plot object
    # Here I'm using i as a proxy for time
    Poincare_sec_phase = i #The total number of frames is 99, so i maps directly onto psi_0 (Poincare-sec-phase-angle)
    # At what angular phase value should I take the poincare section? In units of steps-around-orbital-loop (i.e. 0-99)
    
    # Check my animation by using parameters as set in http://www.scholarpedia.org/article/Duffing_oscillator:
#     const_gamma = 0.3
#     const_delta = 0.2
#     const_beta = 1.0
#     const_alpha = -1.0
#     driving_freq = 1.0 
    const_gamma = 7.5
    const_delta = 0.05
    const_beta = 1.0
    const_alpha = 0.0
    driving_freq = 1.0 


    # Initial conditions (somewhat arbitrary, just followed Merritt's example)
    v0 = 0.0
    x0 = 1.0
    
    skip = 100 # don't plot these , let transient die off. The units of skip is in terms of number-of-orbits
    steps_per_orbit = 100 # Number of steps per orbit
    N = 500 #Number of drive orbits to caluculate for
    max_index = ((N-skip)*steps_per_orbit) - Poincare_sec_phase # Don't try going past here 
                                                            #  i.e. ensure have implemented an upper boundary so don't get an out-of-range error for time interval 
                                                            # (as max_index = N*steps_per_orbit results in out-of-bounds error for non-zero skip,Ponicare_sec_phase)
    timestep = (2*np.pi / driving_freq) / steps_per_orbit # One orbital period = 2pi/omega
    # Create an array of time values that star at t=0, 
    # and end at t=N*orbital_period (where orb_period = 2pi/omega), 
    # and let the difference between the array elements be a timestep
    time_arr = np.arange(0.0, N*(2*np.pi/driving_freq), timestep)
    #plot(LL,style=point,symbol=box,symbolsize=4,axes=boxed,view=[-4..4,-4..4],title="Poincare Section");


    #Calculate solutions to the Duffing oscillator system of two 1st order ODEs
        # scipy.integrate.odeint returns : Array containing the value of y for each desired time in t, with the initial value y0 in the first row
    sol_x_v = integrate.odeint(solvr, [x0, v0], time_arr, args=(const_gamma, const_delta, const_beta, const_alpha, driving_freq)) # Need to pass additional arguments as a tuple
    sol_x_v = sol_x_v[skip*steps_per_orbit:] # chop off initial transient stage
    
    #Initialise lists to store Poicare section state variables (displacement, velocity)
    Poincare_x = []
    Poincare_v = []
    
    # Extract out the [x,v] values at the Poincare-section-phase-angle's, i.e psi_0's, value for each orbit:
    for j in np.arange(Poincare_sec_phase, max_index, steps_per_orbit): #Note the spacing between two adjacent i values = number of steps per orbit
        #print(i) # The index i counts through the orbits (i=0,100,200,300) for Poincare_sec_phase=0
                    # or (i=23,123,223 for Poincare_sec_phase=23). Given that there's a 100 steps per orbit and Poincare_sec_phase is given in units of step_per_orbit
        Poincare_x.append(sol_x_v[j,0])
        Poincare_v.append(sol_x_v[j,1])

    Poincare_sec.set_data(Poincare_x, Poincare_v)
    frame_ctr_text.set_text('frame %d: $\psi_0$=%.2f rads' %(i, i*2*np.pi/steps_per_orbit)) #label frame

    return Poincare_sec, 



#Sort out plots:
fig, ax = plt.subplots(figsize=(6,6))
Poincare_sec, = ax.plot([],[],"k.", markersize=5) 

#ax.plot(Poincare_x, Poincare_v,"r.")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
frame_ctr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) #Label each frame

#ax.set_xlim([-1.5,1.5])
#ax.set_ylim([-1,1])
ax.set_xlim([-3,3])
ax.set_ylim([-6,6])
anim = animation.FuncAnimation(fig, _update_plot, fargs = (Poincare_sec,), 
                               frames = 99, interval = 40, repeat=True, blit=True) #The blit keyword is an important one: this tells the animation to only re-draw the pieces of the plot which have changed

#plt.show()
#FFwriter = animation.FFMpegWriter(fps=10,bitrate=5000,extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])

#anim.save('./Duffing_oscillator_example.mp4')
