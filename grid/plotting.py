import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from grid.policy import get_cell_i, get_cell_j, get_policy_action, get_policy

def _action_arrow(action: str, length: float):

    dx, dy = 0.0, 0.0

    if action == 'U': dx, dy = 0, length
    if action == 'D': dx, dy = 0, -length
    if action == 'L': dx, dy = -length, 0
    if action == 'R': dx, dy = length, 0        
        
    return dx, dy

def plot_policy(policy: dict, states: list, actions: list, m: int, n: int, initial: int, final: int, obstacles: list=[], 
                title: str='', cell_unit_length: float=1.5, action_arrow_length_factor: float=0.2, action_arrow_width_factor: float=0.03):

    a = len(actions)
    A = [*range(a)]
    
    hatches = lambda s: '/' if s in obstacles else ''
    
    action_arrow_length = action_arrow_length_factor*cell_unit_length
    action_arrow_width = action_arrow_width_factor*cell_unit_length

    fig, axs = plt.subplots(1,1,figsize=(n*cell_unit_length,m*cell_unit_length))

    for s in states:

        coord = np.atleast_2d([get_cell_j(s,m,n)+.5,m-get_cell_i(s,m,n)-.5])
        if policy:
            action = get_policy_action(policy,s,actions)[0]

        if s==final:
            axs.scatter(coord[:,0],coord[:,1],s=200,color='red',marker='X')
        elif s==initial:
            axs.scatter(coord[:,0],coord[:,1],s=200,color='blue',marker='X')
        else:
            axs.scatter(coord[:,0],coord[:,1],s=50,color='black')

        if policy:
            if action != 'N':
                
                dx, dy = _action_arrow(action, action_arrow_length)
                axs.arrow(get_cell_j(s,m,n)+.5,
                        m-get_cell_i(s,m,n)-.5,
                        dx,dy,width=action_arrow_width)

    rects = [patches.Rectangle((j,m-i-1),1,1,fill=False,hatch=hatches(i*n+j)) \
                                                     for j in range(n) for i in range(m)]
    for rect in rects:
        axs.add_patch(rect)

    for i in range(m):
        axs.text(-.4,m-i-.5,'{}'.format(i),fontsize=14)

    for j in range(n):
        axs.text(j+.5,m+.4,'{}'.format(j),fontsize=14)
        
    # axs.text(-0.8,m/2,r'$i$',fontsize=20)
    # axs.text(n/2,m+0.7,r'$j$',fontsize=20)

    axs.set_xlim(-1,n+1)
    axs.set_ylim(-1,m+1)    
    axs.tick_params(bottom=False,left=False)
    axs.set(xticklabels=[])
    axs.set(yticklabels=[])
    axs.set_title(title,fontsize=20)

    plt.tight_layout()    

def plot_occupation_measure(rho: np.array, states: list, actions: list, m: int, n: int,
                            title: str='', cell_unit_length: float=1.5, tol: float=1e-4):

    state_num = len(states)
    atol = abs(int(np.log10(tol)))
    policy = get_policy(np.around(rho,atol),states,actions)
    
    _, axs = plt.subplots(1,1,figsize=(n*cell_unit_length,m*cell_unit_length))

    for s in states:

        action = get_policy_action(policy,s,actions)[0]
        a = actions.index(action)

        axs.text(get_cell_j(s,m,n)+.1,m-get_cell_i(s,m,n)-.5,'{:.3f}'.format(rho[state_num*a+s]),fontsize=14)

    rects = [patches.Rectangle((j,m-i-1),1,1,fill=False) for j in range(n) for i in range(m)]

    for rect in rects:
        axs.add_patch(rect)

    axs.set_xlim(-1,n+1)
    axs.set_ylim(-1,m+1)    
    axs.tick_params(bottom=False,left=False)
    axs.set(xticklabels=[])
    axs.set(yticklabels=[])
    axs.set_title(title,fontsize=20)

    plt.tight_layout()    

def plot_reward(risks: list, rewards: list, atol: int=2, set_ticks: bool=True, figsize: tuple=(12,4)):
    
    xs = np.array(risks)
    ys = np.array(rewards)
    n = xs.size
               
    if n>1:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        for i in range(n-1):
            ax.plot(xs[i:i+2], ys[i:i+2],c='b',alpha=0.5, linewidth=3)

        ax.set_xlabel(r'$\alpha$', fontsize=16)
        ax.set_ylabel(r'$r^T \rho_*(\alpha)$', fontsize=16)

        major_ticks = np.unique(np.around(xs,atol))
        ax.tick_params(axis="x", rotation=90, labelsize=12)
        if set_ticks:
            ax.set_xticks(major_ticks)
        ax.grid(which='major', axis='x')
        ax.grid(which='major', alpha=0.6)

        plt.title(r'Reward as a Function of Risk $\alpha$',fontsize=18)

        plt.show()

def plot_ratio(risks: list, rewards: list, atol: int=2, set_ticks: bool=True, figsize: tuple=(12,4)):
    
    xs = np.array(risks)
    ys = np.array(rewards) / np.array(risks)
    n = xs.size
               
    if n>1:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        for i in range(n-1):
            ax.plot(xs[i:i+2], ys[i:i+2],c='b',alpha=0.5, linewidth=3)

        ax.set_xlabel(r'$\alpha$', fontsize=16)
        ax.set_ylabel(r'$r^T \rho_*(\alpha) \,/\, d^T \rho_*(\alpha)$', fontsize=16)

        major_ticks = np.unique(np.around(xs,atol))
        ax.tick_params(axis="x", rotation=90, labelsize=12)
        if set_ticks:
            ax.set_xticks(major_ticks)
        ax.grid(which='major', axis='x')
        ax.grid(which='major', alpha=0.6)


        plt.title(r'Ratio as a Function of Risk $\alpha$',fontsize=18)

        plt.show()