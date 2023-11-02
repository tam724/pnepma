## TODO

# function metropolis_hastings(m0, proposal, target_pdf, N, callback=nothing)
#     x = np.zeros((len(x0), N))
#     accepted = np.full(N, False, dtype=bool)
#     x[:, 0] = x0
#     accepted[0] = True
#     target_pdf_x_i_1 = target_pdf(x[:, 0])
#     for i in range(1, N):
#         # propose
#         x_cand = proposal.sample(x[:, i-1])
#         # acceptance probability  
#         target_pdf_x_cand = target_pdf(x_cand)
#         num = proposal.pdf(x[:, i-1], x_cand)*target_pdf_x_cand
#         denom = proposal.pdf(x_cand, x[:, i-1])*target_pdf_x_i_1
#         if num < denom:
#             alpha = num/denom
#         else:
#             alpha = 1.
#         # sample from uniform
#         u = np.random.rand(1)
#         accepted[i] = u < alpha
#         if accepted[i]:
#             # accept
#             x[:, i] = x_cand
#             target_pdf_x_i_1 = target_pdf_x_cand
#         else:
#             # reject
#             x[:, i] = x[:, i-1]
#             target_pdf_x_i_1 = target_pdf_x_i_1 # basically do nothing here..
#         if callback: callback(x[:, 0:i], accepted[0:i])
#     return x, accepted