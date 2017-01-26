import pstats
p = pstats.Stats('crestats.txt')
#p.strip_dirs().sort_stats(-1).print_stats()


p.sort_stats('tottime').print_stats(30)
