# transfer_config = {
#     'base_classifier': base_classifier,
#     'data_path': data_path
#     'max_depth': max_depth
#     'criterion': criterion,
#     'no_adv_images': no_adv_images,
#     'n_estimators': n_estimators,
#     'epsilons': epsilons,
#     'inner_iter': inner_iter,
#     'outer_iter': outer_iter,
#     'no_threads': no_threads
# }
# # 7 * (7 * 9) ~=~ 500 accuracy calculations
# for est_i in n_estimators:
#     for est_j in n_estimators:
#         for eps in epsilons:
#             adv_transfer(config = transfer_config, clf = est_i, data = est_j, epsilon = eps)


# #Training script

# epsilon = 0.3