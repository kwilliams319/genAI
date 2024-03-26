import numpy as np
import matplotlib.pyplot as plt
import os

def gen_data():
    samples = 50
    th = np.linspace(0, 2*np.pi, samples)[:, None]
    r = np.linspace(-1, 1, 10)

    x1, x2 = r*np.cos(th)+ 1.5, r*np.sin(th)+ 1.5
    X = np.stack([x1, x2], axis=1)

    # x, y = X[:, 0].T, X[:, 1].T
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
    # plt.plot(x, y, 'o')
    # plt.plot(0, 0, 'ko')
    # plt.show()

    steps = 200

    beta_schedule = np.linspace(1/steps, 1, steps-1)

    X_data = []
    N_data = []

    for i in range(steps-1):

        Bt = beta_schedule[i]

        X_ = np.random.normal((1-Bt*.03)*X, Bt/4)

        # xy_data = np.concatenate((xy_data, np.array([[i/200.]*50]).T), axis=1)

        
        X_data.append(X_.reshape(samples, 20))
        N_data.append((X_ - X).reshape(samples, 20))


        X = X_

        if i%4 == 0:

            # x, y = X[[0, 12], 0].T, X[[0, 12], 1].T
            x, y = X[:, 0].T, X[:, 1].T
            plt.cla()
            # plt.axis('equal')
            plt.plot(x, y, 'o')
            plt.plot(np.mean(X[:, 0]), np.mean(X[:, 1]), 'ko')
            plt.plot(0, 0, 'ko')
            # plt.xlim([-1, 3])
            # plt.ylim([-1, 3])
            plt.xlim([-4, 4])
            plt.ylim([-4, 4])
            plt.title(i)
            plt.pause(.1)

            # Save figure
            plt.savefig(os.path.join('figures', f'timestep_{i}.png'))


        # std, mean = np.std(X), np.mean(X)
        # print(mean, std)

    plt.show()

    X_data = np.array(X_data).reshape(-1, 20)
    N_data = np.array(N_data).reshape(-1, 20)

    # np.save('X_.npy', X_data)
    # np.save('N.npy', N_data)

gen_data()


