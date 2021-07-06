import torch
ans = {'values':{'life_datas':torch.randn(2,3).cuda(),
                 'hand_card_costs': [torch.randn(2,1).cuda() for i in range(9)],
                 'follower_stats': [torch.randn(2,2).cuda() for i in range(10)],
                 'pp_datas':torch.randn(2,2).cuda(),
                 'able_to_play':torch.randint(9, (2, 9)).cuda(),
                 'able_to_attack':torch.randint(5, (2, 5)).cuda(),
                 'able_to_creature_attack':torch.randint(5, (2, 5)).cuda(),
                 'one_hot_able_actions':torch.randn(2,25).cuda()
                 },
       'hand_ids': [torch.randint(1000, (2, 1)).cuda() for i in range(9)],
       'follower_card_ids': [torch.randint(100, (2, 1)).cuda() for i in range(10)],
       'amulet_card_ids': [torch.randint(100, (2, 1)).cuda() for i in range(10)],
       'follower_abilities':[torch.randint(15, (2, 16)).cuda() for i in range(10)],
       'able_to_evo':torch.randint(5, (2, 5)).cuda()}

action_codes_dict = {'action_categories': [torch.randint(4,(2,1)).cuda() for i in range(25)],
                     'card_locations': [torch.randint(4,(2,1)).cuda() for i in range(25)],
                     'card_ids': [torch.randint(3000,(2,1)).cuda() for i in range(25)],
                     'able_to_choice': torch.ones(2,25).cuda()}

ans['detailed_action_codes'] =action_codes_dict

action = torch.randint(0,24,(2,1)).cuda()
rewards = torch.Tensor([[1.0],[-1.0]]).cuda()