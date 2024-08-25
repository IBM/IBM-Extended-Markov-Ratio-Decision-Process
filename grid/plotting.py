from typing import Optional, Union, List, Callable
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

from grid.policy import get_cell_i, get_cell_j, get_policy_action, get_policy

def _action_arrow(action: str, length: float):

    dx, dy = 0.0, 0.0

    if action == 'U': dx, dy = 0, length
    if action == 'D': dx, dy = 0, -length
    if action == 'L': dx, dy = -length, 0
    if action == 'R': dx, dy = length, 0        
        
    return dx, dy

def _plot_single_grid(m: int, n: int, initial: int, final: int, states: List, actions: List, policy: dict, 
                      axs: Axes, title: Optional[str], title_fontsize: int,
                      switch_state: Optional[int], switch_action: Optional[int], switch_alpha: float, 
                      hatches: Callable[[int], str], action_arrow_length: float, action_arrow_width: float):
    
    for s in states:

        coord = np.atleast_2d([get_cell_j(s,m,n)+.5,m-get_cell_i(s,m,n)-.5])
        
        if policy:
            action = get_policy_action(policy,s,actions)[0]

        if s==final:
            axs.scatter(coord[:,0],coord[:,1],s=200,color='red',marker='X',alpha=switch_alpha)
        elif s==initial:
            axs.scatter(coord[:,0],coord[:,1],s=200,color='blue',marker='X',alpha=switch_alpha)
        else:
            if s==switch_state:
                axs.scatter(coord[:,0],coord[:,1],s=50,color='black')
            else:
                axs.scatter(coord[:,0],coord[:,1],s=50,color='black',alpha=switch_alpha)

        if policy:
            if action != 'N':
                if s==switch_state:
                    dx, dy = _action_arrow(switch_action, action_arrow_length)
                    axs.arrow(get_cell_j(s,m,n)+.5,m-get_cell_i(s,m,n)-.5,dx,dy,width=action_arrow_width)
                dx, dy = _action_arrow(action, action_arrow_length)
                axs.arrow(get_cell_j(s,m,n)+.5,m-get_cell_i(s,m,n)-.5,dx,dy,width=action_arrow_width,alpha=switch_alpha)

    rects = [patches.Rectangle((j,m-i-1),1,1,fill=False,hatch=hatches(i*n+j)) for j in range(n) for i in range(m)]
    for rect in rects:
        axs.add_patch(rect)

    for i in range(m):
        axs.text(-.4,m-i-.5,'{}'.format(i),fontsize=title_fontsize)

    for j in range(n):
        axs.text(j+.5,m+.4,'{}'.format(j),fontsize=title_fontsize)
        
    axs.set_xlim(-1,n+1)
    axs.set_ylim(-1,m+1)    
    axs.tick_params(bottom=False,left=False)
    axs.set(xticklabels=[])
    axs.set(yticklabels=[])
    if title:
        axs.set_title(title,fontsize=title_fontsize)    
    
    
def plot_policy(policy: Union[dict,List[dict]], states: list, actions: list, m: int, n: int, initial: int, final: int, 
                obstacles: list=[], switch_state: Optional[Union[int,List[int]]]=None, switch_action: Optional[Union[int,List[int]]]=None, 
                title: Optional[Union[str,List[str]]]=None, title_fontsize: int=14, cell_unit_length: float=1.5, action_arrow_length_factor: float=0.2, action_arrow_width_factor: float=0.03, ncols: int=2):

    a = len(actions)
    A = [*range(a)]
    
    hatches = lambda s: '/' if s in obstacles else ''
    
    action_arrow_length = action_arrow_length_factor*cell_unit_length
    action_arrow_width = action_arrow_width_factor*cell_unit_length
    
    assert all([isinstance(policy, dict),isinstance(switch_state, int),isinstance(switch_action, str),isinstance(title, str)]) or\
            all([isinstance(obj, list) for obj in [policy,switch_state,switch_action,title]]) or\
            all([isinstance(policy, list) or isinstance(policy, dict)]+\
                [isinstance(title, list) or (title is None) or isinstance(title, str)]+\
                [obj is None for obj in [switch_state,switch_action]]),\
    'If you plot a list of policies, make sure you provide a list of switch states/actions and titles or leave them as None'
                
    if isinstance(policy, dict):

        switch_alpha = 1.0 if any([switch_state is None, switch_action is None]) else 0.2

        _, axs = plt.subplots(1,1,figsize=(n*cell_unit_length,m*cell_unit_length))

        _plot_single_grid(m, n, initial, final, states, actions, policy, 
                          axs, title, title_fontsize, switch_state, switch_action, switch_alpha, 
                          hatches, action_arrow_length, action_arrow_width)
        
    elif isinstance(policy, list):
        
        nrows = ceil(len(policy)/ncols)
        _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(n*cell_unit_length*ncols,m*cell_unit_length*nrows))
        
        for k, p in enumerate(policy):            
            j = k%ncols
            i = int(k/ncols)
            ttl = title[k] if isinstance(title,list) else None
            sw_state = switch_state[k] if isinstance(switch_state,list) else None
            sw_action = switch_action[k] if isinstance(switch_action,list) else None
            sw_alpha = 1.0 if any([sw_state is None, sw_action is None]) else 0.2
            ax = axs[j] if nrows==1 else axs[i,j]
            _plot_single_grid(m, n, initial, final, states, actions, p, 
                          ax, ttl, title_fontsize, sw_state, sw_action, sw_alpha, 
                          hatches, action_arrow_length, action_arrow_width)
            
        for i in range(nrows):
            for j in range(ncols):
                if (i+1)*(j+1)>k+1:
                    ax = axs[j] if nrows==1 else axs[i,j]
                    ax.axis('off')

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

def plot_ratio(risks: list, rewards: list, atol: int=2, set_ticks: bool=True, mark_min: bool=False, ylabel_weighted: bool=False, figsize: tuple=(12,4), titlefontsize: int=20):
    
    xs = np.array(risks)
    ys = np.array(rewards) / np.array(risks)
    n = xs.size
               
    if n>1:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        if mark_min:
            arg = np.argmin(ys)
            xmin = np.around(xs[arg],atol)

        ax.plot(xs,ys,c='b',marker='o',markersize=3,alpha=0.5, linewidth=3)
        ax.set_xlabel(r'$\alpha$', fontsize=16)
        if ylabel_weighted:
            ax.set_ylabel(r'$r^T \rho(\alpha) \,/\, \sum_{i} \left( d_i^T \rho(\alpha) \right)^{\omega_i}$', fontsize=16)
        else:
            ax.set_ylabel(r'$r^T \rho(\alpha) \,/\, d^T \rho(\alpha)$', fontsize=16)

        major_ticks = np.unique(np.around(xs,atol)).tolist() if set_ticks else np.around(ax.get_xticks(),atol).tolist()[1:]
        ax.tick_params(axis="x", rotation=90, labelsize=12)
        if mark_min:
            ax.set_xticks(major_ticks+[xmin])
            ax.axvline(x=xmin,c='r',alpha=0.4)
        else:
            ax.set_xticks(major_ticks)
            
        ax.grid(which='major', axis='x')
        ax.grid(which='major', alpha=0.6)
        ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{atol}f'))

        plt.title(r'Ratio as a Function of Risk $\alpha$',fontsize=titlefontsize)

        plt.show()