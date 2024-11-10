from hidet.graph.transforms import registered_rewrite_rules, register_resolve_rule, SubgraphRewriteRule


print('Registered rewrite rules:')
for rule in registered_rewrite_rules():
    assert isinstance(rule, SubgraphRewriteRule)
    print(rule.name)
