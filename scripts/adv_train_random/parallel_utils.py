from multiprocessing import Process, Manager
import numpy as np

def accumulate_parallel_function(function, n_boost_random_train_samples, cur_estimator, epsilon, X, y, set_size):

    indexArray = list(range(0,n_boost_random_train_samples,set_size))

    print("X.shape[0]:", X.shape[0])

    with Manager() as manager:
        ListAdvImages = manager.list()
        processes = [] 
        for index in indexArray:
            p = Process(target = function, args = (ListAdvImages, cur_estimator, epsilon, X, y, index, set_size))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        adv_examples = ListAdvImages[0][0]
        adv_y = ListAdvImages[0][1]
        print(adv_examples)
        print(adv_y)
        # print("adv_examples shape:", adv_examples.shape)
        # print("adv_y shape:", adv_y.shape)

        for i in range(1, len(ListAdvImages)):
            # print("ListAdvImages[{}][0] shape:".format(i), ListAdvImages[i][0].shape)
            # print("ListAdvImages[{}][1] shape:".format(i), ListAdvImages[i][1].shape)

            adv_examples = np.concatenate((adv_examples, ListAdvImages[i][0]))
            adv_y = np.concatenate((adv_y, ListAdvImages[i][1]))
        return adv_examples, adv_y
