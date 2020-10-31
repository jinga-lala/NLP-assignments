from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
from nltk.tag.mapping import tagset_mapping
from anytree import Node, RenderTree
import nltk

PTB_UNIVERSAL_MAP = tagset_mapping('en-ptb', 'universal')
parser = CoreNLPParser(url='http://localhost:9000')

    
def cp(s): 
	return list(parser.raw_parse(s))

def Include_Dependency(NP_list,Max_index):

	Head_Node = NP_list[Max_index]
	if Max_index>0 and (NP_list[Max_index-1].name.split()[0][:2] == 'NN' or NP_list[Max_index-1].name.split()[0][:2] == 'VB'):
		COMP = Node("COMPOUND")
		Dummy_Node = Node("Dummy")
		for child in NP_list[Max_index].children:
			child.parent = Dummy_Node

		NP_list[Max_index-1].parent = COMP
		COMP.parent = NP_list[Max_index]
		for child in Dummy_Node.children:
			child.parent = NP_list[Max_index]
		NP_list.pop(Max_index-1)
		Head_Node = Include_Dependency(NP_list,Max_index-1)

	elif Max_index>0 and NP_list[Max_index-1].name.split()[0][:2] == 'JJ':
		AMOD = Node("AMOD")
		Dummy_Node = Node("Dummy")
		for child in NP_list[Max_index].children:
			child.parent = Dummy_Node

		NP_list[Max_index-1].parent = AMOD
		AMOD.parent = NP_list[Max_index]
		for child in Dummy_Node.children:
			child.parent = NP_list[Max_index]
		NP_list.pop(Max_index-1)
		Head_Node = Include_Dependency(NP_list,Max_index-1)

	elif Max_index>0 and NP_list[Max_index-1].name.split()[0] == 'DT':
		AMOD = Node("DET")
		Dummy_Node = Node("Dummy")
		for child in NP_list[Max_index].children:
			child.parent = Dummy_Node

		NP_list[Max_index-1].parent = AMOD
		AMOD.parent = NP_list[Max_index]
		for child in Dummy_Node.children:
			child.parent = NP_list[Max_index]
		NP_list.pop(Max_index-1)
		Head_Node = Include_Dependency(NP_list,Max_index-1)

	elif Max_index>0 and NP_list[Max_index-1].name.split()[0] == 'PRP$':
		AMOD = Node("POSS")
		Dummy_Node = Node("Dummy")
		for child in NP_list[Max_index].children:
			child.parent = Dummy_Node

		NP_list[Max_index-1].parent = AMOD
		AMOD.parent = NP_list[Max_index]
		for child in Dummy_Node.children:
			child.parent = NP_list[Max_index]
		NP_list.pop(Max_index-1)
		Head_Node = Include_Dependency(NP_list,Max_index-1)

	else:
		Head_Node = NP_list[Max_index]
		for subtree in range(0,len(NP_list)):

			if subtree != Max_index:
				NP_list[subtree].parent = Head_Node


	return Head_Node

def PP_subtree(tree):
	
	Head_Node = Node(tree[0].label()+" "+tree[0][0])
	del tree[tree[0].treepositions()]
	S_Bar = Con_to_Dep(tree)
	S_Bar.name = "PCOMP"
	S_Bar.parent = Head_Node

	return Head_Node

def ADVP_subtree(tree):
	Head_Node = None
	if len(tree) > 1:
		pass

	else:
		Head_Node = Node(tree[0].label()+" "+tree[0][0])

	return Head_Node

def NP_subtree(tree):

	Max_value = -1
	Sub_NP_NX_exist = 0
	NP_NX_list = []
	for subtree in tree:
		
		if subtree.label() == 'NP' or subtree.label() == 'NX' or subtree.label() == 'WHNP':
			Sub_NP_NX_exist = 1
			output = NP_subtree(subtree)
			NP_NX_list.append((output[0],output[1]))

		elif subtree.label() == 'ADJP':
			pass

		elif subtree.label() == 'SBAR':
			RELCL_value = Con_to_Dep(subtree[1]).children[0]
			RELCL = Node("RELCL")
			VERB_temp = Node("",parent=RELCL)
			VERB_temp.name = RELCL_value.name 
			NSUBJ_value = NP_subtree(subtree[0])[1]
			NSUBJ = Node("NSUBJ",parent=VERB_temp)
			NSUBJ_value.parent = NSUBJ
			for child in RELCL_value.children:
				child.parent = VERB_temp

			NP_NX_list.append(RELCL)

		else:
			New_Node = Node(subtree.label()+" "+subtree[0])
			NP_NX_list.append(New_Node)

	if Sub_NP_NX_exist == 1:		
		Max_index = 0
		Final_List = []
		for item in range(0,len(NP_NX_list)):

			if isinstance(NP_NX_list[item], tuple):

				if NP_NX_list[item][0]>Max_value:
					Max_value = NP_NX_list[item][0]
					Max_index = item
				Final_List.append(NP_NX_list[item][1])

			else:
				Final_List.append(NP_NX_list[item])

		NSUBJ_value = Include_Dependency(Final_List,Max_index)

	else:
		Max_index = 0
		for item in range(0,len(NP_NX_list)):
			if NP_NX_list[item].name.split()[0][:2] == 'NN':
				Max_index = item
	
		NSUBJ_value = Include_Dependency(NP_NX_list,Max_index)


	return (Max_value+1, NSUBJ_value)	


def VP_subtree(tree):

	VERB = Node("VERB")
	
	Sub_VP_exists = 0

	for subtree in tree:
		if subtree.label() == 'VP':
			Sub_VP_exists = 1
			VP_subtree(subtree)

	if Sub_VP_exists == 0:

		##Find main verb
		verb_index = -1
		for subtree in range(0,len(tree)):
			if tree[subtree].label()[:2] == 'VB':
				verb_index = subtree

		Main_Verb = Node(tree[verb_index].label()+" "+tree[verb_index][0],parent=VERB)

		for subtree in range(0,len(tree)):

			if subtree == verb_index:
				pass

			elif tree[subtree].label() == 'PP':
				PREP = Node("PREP",parent=VERB.children[0])
				PREP_value = PP_subtree(tree[subtree])
				PREP_value.parent = PREP

			elif tree[subtree].label()[:2] == 'NP':
				DOBJ = Node("DOBJ",parent=VERB.children[0])
				DOBJ_value = NP_subtree(tree[subtree])[1]
				DOBJ_value.parent = DOBJ

			elif tree[subtree].label() == 'ADVP':
				ADVMOD = Node("ADVMOD",parent=VERB.children[0])
				ADVMOD_value = ADVP_subtree(tree[subtree])
				ADVMOD_value.parent = ADVMOD


	else:
		pass

	return VERB



def Con_to_Dep(tree):

	print(tree)
	ROOT = Node("ROOT")
	VERB = Node("Not filled",parent=ROOT)
	
	for subtree in tree[0]:

		if subtree.label() == 'VP':
			VERB_temp = VP_subtree(subtree)
			VERB.name = VERB_temp.children[0].name
			while len(VERB_temp.children[0].children) > 0:
				VERB_temp.children[0].children[0].parent = VERB


		if subtree.label()[:2] == 'NP' or subtree.label() == 'WHNP':
			NSUBJ = Node("NSUBJ",parent=VERB)
			NSUBJ_value = NP_subtree(subtree)[1]
			NSUBJ_value.parent = NSUBJ

		if subtree.label() == '.':
			PUNCT = Node("PUNCT",parent=VERB)
			PUNCT_value = Node(subtree.label()+" "+subtree[0])
			PUNCT_value.parent = PUNCT

	return ROOT


if __name__ == '__main__':

	sentence = "Senior students who had finished their exams played energetically street football with crowd watching."

	doc = cp(sentence)
	print(doc)
	ROOT = Con_to_Dep(doc[0])

	for pre, _, node in RenderTree(ROOT):
		print("%s%s" % (pre, node.name))