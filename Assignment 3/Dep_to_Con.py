import spacy
from anytree import Node, RenderTree



def punct_subtree(tree):

	PUNCT = Node(tree.text+" "+tree.tag_)
	
	return PUNCT


def appos_subtree(tree):

	return NP_subtree(tree)



def advmod_subtree(tree):

	ADVP = Node("ADVP")
	
	Head_ADV = Node(tree.text+" "+tree.tag_,parent=ADVP)
	
	if tree.n_rights > 0:

		for token in tree.rights:

			if token.dep_ == 'cc':
				CC_value = Node(token.text+" "+token.tag_,parent=ADVP)

			if token.dep_ == 'conj':
				Conj_value = advmod_subtree(token)
				if len(Conj_value.children) > 1:
					Conj_value.parent = ADVP
				else:
					Conj_value.children[0].parent = ADVP

	return ADVP


def amod_subtree(tree):

	ADJP = Node("ADJP")

	if tree.n_lefts > 0:

		for token in tree.lefts:

			if token.dep_ == 'advmod':
				ADVMOD_value = Node(token.text+" "+token.tag_,parent=ADJP)

	
	Head_ADJ = Node(tree.text+" "+tree.tag_,parent=ADJP)
	
	if tree.n_rights > 0:

		for token in tree.rights:

			if token.dep_ == 'cc':
				CC_value = Node(token.text+" "+token.tag_,parent=ADJP)

			if token.dep_ == 'conj':
				Conj_value = amod_subtree(token)
				if len(Conj_value.children) > 1:
					Conj_value.parent = ADJP
				else:
					Conj_value.children[0].parent = ADJP

	return ADJP


def acomp_subtree(tree):

	##Adjective Complement	
	ADJP = Node("ADJP")
	Head_ADJ = Node(tree.text+" "+tree.tag_,parent=ADJP)

	if tree.n_rights > 0:

		for token in tree.rights:

			if token.dep_ == 'cc':
				CC_value = Node(token.text+" "+token.tag_,parent=ADJP)

			if token.dep_ == 'conj':

				Conj_value = acomp_subtree(token)
				if len(Conj_value.children) > 1:
					Conj_value.parent = ADJP
				else:
					Conj_value.children[0].parent = ADJP


	return ADJP


def NP_subtree(tree):

	NP = Node("NP")
	# Head_NP = Node(root_node.text+" "+root_node.tag_,parent=NP)

	if tree.n_lefts > 0:

		for token in tree.lefts:

			if token.dep_ == 'det':
				det_Value = Node(token.text+" "+token.tag_,parent=NP)

			if token.dep_ == 'poss':
				det_Value = Node(token.text+" "+token.tag_,parent=NP)

			if token.dep_ == 'amod':
				ADJP_value = amod_subtree(token)
				if len(ADJP_value.children) > 1:
					ADJP_value.parent = NP
				else:
					ADJP_value.children[0].parent = NP

			if token.dep_ == 'compound':
				Compound_Value = Node(token.text+" "+token.tag_,parent=NP)



	
	Head_NP = Node(tree.text+" "+tree.tag_,parent=NP)
	
	if tree.n_rights > 0:

		for token in tree.rights:

			if token.dep_ == 'cc':
				CC_value = Node(token.text+" "+token.tag_,parent=NP)

			if token.dep_ == 'conj':
				Conj_value = NP_subtree(token)

				if len(Conj_value.children) <= 1:
					Conj_value.children[0].parent = NP
				####Check if subtrees joined by CONJ
				else:
					Conj_joined = 0
					for child in Conj_value.children:
						if child.name.split()[1] == 'CC':
							Conj_joined = 1

					if Conj_joined == 1:
						while len(Conj_value.children) > 0:
							Conj_value.children[0].parent = NP

					else:
						Conj_value.parent = NP

			if token.dep_ == 'punct':
				Punct_value = punct_subtree(token)
				Punct_value.parent = NP

			if token.dep_ == 'appos':
				Appos_value = appos_subtree(token)
				Appos_value.parent = NP

	return NP


def prep_subtree(tree):

	PP = Node("PP")

	Prep_Root = Node(tree.text+" "+tree.tag_,parent=PP)

	if tree.n_rights > 0:

		for token in tree.rights:	

			if token.dep_ == 'pcomp':
				S_bar = Dep_to_Con(token)
				S_bar.parent = PP

	return PP

def modify_VP_tree(tree):

	modification = 0
	index_vp = 0
	while index_vp < len(tree.children):
		if tree.children[index_vp].name == 'VP':

			if index_vp+1<len(tree.children) and tree.children[index_vp+1].name == 'ADVP':
				modification = 1
				tree.children[index_vp+1].parent = tree.children[index_vp]
			
			if index_vp-1>=0 and tree.children[index_vp-1].name == 'ADVP':
				modification = 1
				Dummy_Node = Node("Dummy")
				while len(tree.children[index_vp].children) > 0:
					tree.children[index_vp].children[0].parent = Dummy_Node

				tree.children[index_vp-1].parent = tree.children[index_vp]
				while len(Dummy_Node.children) > 0:
					Dummy_Node.children[0].parent = tree.children[index_vp-1]

				index_vp-=1


			if index_vp+1<len(tree.children) and tree.children[index_vp+1].name == 'NP':
				modification = 1
				tree.children[index_vp+1].parent = tree.children[index_vp]


			if index_vp-1>=0 and tree.children[index_vp-1].name == 'AUX':
				modification = 1
				Dummy_Node = Node("VP")
				while len(tree.children[index_vp].children) > 0:
					tree.children[index_vp].children[0].parent = Dummy_Node

				tree.children[index_vp-1].parent = tree.children[index_vp]
				Dummy_Node.parent = tree.children[index_vp-1]

				index_vp-=1

			if index_vp+1<len(tree.children) and tree.children[index_vp+1].name == 'ADVP':
				modification = 1
				tree.children[index_vp+1].parent = tree.children[index_vp]

			if index_vp+1<len(tree.children) and tree.children[index_vp+1].name == 'ADVP':
				modification = 1
				tree.children[index_vp+1].parent = tree.children[index_vp]

		index_vp+=1

	if modification == 1:
		tree = modify_VP_tree(tree)

	while len(tree.children) ==1 and tree.children[0].name == 'VP':
		tree = tree.children[0] 

	return tree


def Dep_to_Con(tree):


	S = Node("S")
	VP = Node("VP")

	if tree.n_lefts > 0:

		###Check if first AUX or NOT
		index = 0

		for token in tree.lefts:

			if token.dep_ == 'aux' and index == 0:
				AUX = Node('AUX',parent=S)
				Aux_Value = Node(token.text+" "+token.tag_,parent=AUX)

			if token.dep_ == 'nsubj':
				NP = NP_subtree(token)
				NP.parent = S

			if (token.dep_ == 'aux' or token.dep_ == 'auxpass') and index != 0:
				AUX = Node('AUX',parent=VP)
				Aux_VP = Node(token.text+" "+token.tag_,parent=AUX)

			if token.dep_ == 'advmod':
				ADVP = Node("ADVP",parent=VP)
				ADVP_value = Node(token.text+" "+token.tag_,parent=ADVP)

			index+=1


	VP_verb = Node("VP",parent=VP)
	Verb_Root = Node(tree.text+" "+tree.tag_,parent=VP_verb)
	
	if tree.n_rights > 0:

		for token in tree.rights:

			if token.dep_ == 'acomp':
				ADJP_value = acomp_subtree(token)
				ADJP_value.parent = VP

			if token.dep_ == 'advmod':
				ADVP = advmod_subtree(token)
				ADVP.parent = VP

			if token.dep_ == 'dobj':
				NP_OBJ = NP_subtree(token)
				NP_OBJ.parent = VP

			if token.dep_ == 'dative' or token.dep_ == 'iobj':
				NP_OBJ = NP_subtree(token)
				NP_OBJ.parent = VP

			if token.dep_ == 'npadvmod':
				NP_OBJ = NP_subtree(token)
				NP_OBJ.parent = VP

			if token.dep_ == 'prep':
				PP = prep_subtree(token)
				PP.parent = VP

			if token.dep_ == 'punct':
				Punct_value = punct_subtree(token)
				Punct_value.parent = VP



	VP = modify_VP_tree(VP)

	if VP.children[len(VP.children)-1].name == ". .":
		VP.parent = S
		VP.children[len(VP.children)-1].parent = S
	

	return S



if __name__ == '__main__':

	nlp = spacy.load("en_core_web_sm")
	sentence = ("Ram, Shyam and Mohan are friends.")

	print("\n\n\nPrinting Dependency Tree---------------------")

	doc = nlp(sentence)
	for token in doc:
	     print(token.text, token.tag_, token.dep_, [child for child in token.children])

	doc = list(doc.sents)[0]

	S = Dep_to_Con(doc.root)

	for pre, _, node in RenderTree(S):
		print("%s%s" % (pre, node.name))



