# -*- coding: utf-8 -*-
"""
This file is based on the function tree2branches() on preprocessing.py. tree2branches() only produce every branch in a conversation as a list, not take the conversation as a whole structure. 
"""

def Branches2

def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    while True:
        node_name = node.keys()[i]
        #print node_name
        branch.append(node_name)
        # get children of the node
        first_child = node.values()[i]
        # actually all chldren, all tree left under this node
        if first_child != []:  # if node has children
            node = first_child      # walk down
            parent_tracker.append(node)
            siblings = first_child.keys()
            i = 0  # index of a current node
        else:
            branches.append(deepcopy(branch))
            i = siblings.index(node_name)  # index of a current node
            # if the node doesn't have next siblings
            while i+1 >= len(siblings):
                if node is parent_tracker[0]:  # if it is a root node
                    return branches
                del parent_tracker[-1]
                del branch[-1]
                node = parent_tracker[-1]      # walk up ... one step
                node_name = branch[-1]
                siblings = node.keys()
                i = siblings.index(node_name)
            i = i+1    # ... walk right
#            node =  parent_tracker[-1].values()[i]
            del branch[-1]
#            branch.append(node.keys()[0])
#%%
# process tweet into features

