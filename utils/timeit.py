
    # if args.do_test:
    #     logging.info('Evaluating on Test Dataset...')
    #     # warmup
    #     for _ in range(5):
    #         # _ = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #         # _ = evaluate(inductor_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
    #         _ = evaluate(hidet_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
        
    #     # Torch Profiler
    #     from torch.profiler import profile, record_function, ProfilerActivity
    #     # with torch.autograd.profiler.profile(enabled=True) as prof:
    #     # with torch.autograd.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     # torch.cuda.memory._record_memory_history()
    #     # test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #     # test_all_metrics = evaluate(inductor_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #     test_all_metrics = evaluate(hidet_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #     # torch.cuda.memory._dump_snapshot("hidet.pickle")
    #     # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #     # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        

    #     # Timeit: Profile the speed up of hidet and inductor
    #     import timeit
    #     NUM_ITERS=50
    #     # with torch.no_grad():
    #         # warmup
    #         # for _ in range(10):
    #         #     _ = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #         #     _ = evaluate(inductor_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
    #         #     _ = evaluate(hidet_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
            
    #         # eager_time, inductor_time, hidet_time = 0, 0, 0
    #         # for i in range(10):
    #         #     starttime = timeit.default_timer()
    #         #     _ = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)    
    #         #     endtime = timeit.default_timer()
    #         #     eager_time += endtime - starttime
    #         #     starttime = timeit.default_timer()
    #         #     _ = evaluate(inductor_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
    #         #     endtime = timeit.default_timer()
    #         #     inductor_time += endtime - starttime
    #         #     starttime = timeit.default_timer()
    #         #     _ = evaluate(hidet_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
    #         #     endtime = timeit.default_timer()
    #         #     hidet_time += endtime - starttime

    #         # par = (model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)
    #         # eager_time = timeit.timeit("evaluate(*par)", number=NUM_ITERS, globals=globals())
    #         # eager_time = timeit.timeit("evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)", setup='from __main__ import evaluate',number=NUM_ITERS, globals=globals())
    #         # inductor_t = timeit.timeit("evaluate(inductor_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)", setup='from __main__ import evaluate',number=NUM_ITERS, globals=globals())
    #         # hidet_t = timeit.timeit("evaluate(hidet_model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)", setup='from __main__ import evaluate', number=NUM_ITERS, globals=globals())
            
    #     # print(f"eager use: {eager_time * 1000 / NUM_ITERS} ms/iter")
    #     # print(f"inductor use: {inductor_time * 1000 / NUM_ITERS} ms/iter")
    #     # print(f"hidet use: {hidet_time * 1000 / NUM_ITERS} ms/iter")
    #     # print(f"inductor speed up ratio: {eager_time / inductor_time}")
    #     # print(f"hidet speed up ratio: {eager_time / hidet_time}")


    # # logging.info("Finished!!")
