import numpy as np
import matplotlib as plt

def pbil(F, d, N, th1, th2, th3, steps=500):  
    
    plot_probs = np.zeros((steps, d))
    plot_scores = np.zeros(steps)
    
    p = np.ones((d, 1)) / 2
    population = np.random.rand(d, N) < p
    result = F(population)
      
    plot_probs[0] = p.reshape(-1)
    plot_scores[0] = np.max(result)
    
    for i in range(1, steps):
        
        best = population[:, [np.argmax(result)]]
        p = p * (1 - th1) + best * th1
        
        if_change = np.random.rand(*(d,1)) < th2
        p = p * (1 - if_change) + (
            (if_change) * (
                p * (1 - th3)
                + (np.random.rand(*(d,1)) < 0.5) * th3
            )
        )
        p = p.clip(0,1)
        population = np.random.rand(d, N) < p
        result = F(population)
        
        plot_probs[i] = p.reshape(-1)
        plot_scores[i] = np.max(result)
    
    return plot_scores, plot_probs
        
def plot_results(scores, probs, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)
    for i in range(scores.shape[1]):
        axs[0].plot(range(scores.shape[0]), scores[:, i])
    
    for i in range(probs.shape[1]):
        axs[1].plot(range(probs.shape[0]), probs[:,i,0])
    plt.show()