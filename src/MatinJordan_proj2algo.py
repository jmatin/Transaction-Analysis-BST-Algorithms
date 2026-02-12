import sys
class Stack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def top(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)
class Queue:
    def __init__(self):
        self.items = []
    def head(self):
        return self.items[len(self.items) - 1]
    def isEmpty(self):
        return self.items == []
    def insert(self, item):
        self.items.insert(0, item)
    def remove(self):
        return self.items.pop()
    def size(self):
        return len(self.items)
class Node :
    def __init__ ( self ,transaction_id, transaction_amount ):
        self.transaction_id = transaction_id
        self.left = None
        self.right = None
        self.father = None
        self.height = 1
        self.transaction_amount = transaction_amount

class BinarySearchTree :
    """Je fais prendre à  BinarySearchTree les deux paramètres d'identification d'une transaction,
    et pour avoir les autres attributs de Node il y aura : '.root' suivi de left,right ou father.
    Ce n'est peut être pas le choix le plus malin, mais pour moi c'est << logique >>
    """
    def __init__(self,transaction_id, transaction_amount):
        self.transaction_amount = transaction_amount
        self.transaction_id = transaction_id
        self.root = Node(transaction_id,transaction_amount)

    def getRootVal(self):
        return self.transaction_amount
    def setRootVal(self, transaction_id,trasaction_amount):
        self.transaction_id = transaction_id
        self.transaction_amount = trasaction_amount
    def getData(self):
        return self.root.transaction_amount
    def getGlobalRoot(self):
        res = self.root
        if res != None:
            while res.father != None:
                res = res.father
        return res
    def getNext(self, node):
        """On a repris le getNext du cours et on lui a passé objet Node en paramètre
        """
        if self.getLast() == node.transaction_amount :  #Clairement, le dernier élément n'a pas de suivant
            return -1
        res = self.root              #Je démarre de la racine
        while res.transaction_amount < node.transaction_amount:      #Je boucle tant que je n'ai pas atteint l'élément
            res = res.right                                          #dont je veux prendre le suivant
        while res.transaction_amount > node.transaction_amount:
            res = res.left
        if res.right != None :                                       #J'applique dans le if et le else qui suivent
            res = res.right                                          #exactement la même logique que dans le code
            while res.left != None:                                  #du cours
                res = res.left
            return res
        else:
            fils = self
            res = self.root.father
            while res != None and res.right == fils:
                fils = res
                res = res.father
        return res
    def getPrevious(self,node):
        """ elle se comporte stricto senso comme sa jumelle getNext dans le sens ou chaque "right est remplacé par un
        left et vice versa
        """
        if self.getFirst().transaction_amount == node.transaction_amount:
            return -1
        res = self.root
        while res.transaction_amount < node.transaction_amount:
            res = res.right
        while res.transaction_amount > node.transaction_amount:
            res = res.left
        if res.left != None:
            res = res.left
            while res.right != None:
                res = res.right
            return res
        else:
            fils = self
            res = self.root.father
            while res != None and res.left == fils:
                fils = res
                res = res.father
        return res
    def getFirst(self):
        """Evidemment, nul besoin de lui passer un objet Node
        """
        res = self.getGlobalRoot()
        while res.left != None:
            res = res.left
        return res
    def getLast(self):
        """Idem
        """
        res = self.getGlobalRoot()
        while res.right != None:
            res = res.right
        return res
    def insert(self, node):
        """Le code suit presque la même logique que celui du cours, il diffère toutefois dans le fait que
           l'on ne rajoute pas un objet BinarySearchTree mais un objet Node.
        """
        res = self.getGlobalRoot()
        while res != None:
            p = res
            if node.transaction_amount < p.transaction_amount:
                res = res.left
            else:
                if isinstance(res ,Node) :
                    res = res.right
                """elif isinstance(res, BinarySearchTree) :
                    res = res.root.right"""
        if node.transaction_amount < p.transaction_amount:
            p.left = Node(node.transaction_id,node.transaction_amount)
            p.left.father = p
        else:
            p.right = Node(node.transaction_id,node.transaction_amount)
            p.right.father = p

    def find(self, key_id):
        """Puisque le numéro d'identification est indépendant de la position de son noeud dans le BST, je vais simplement
           procéder à un parcours d'arbre comme si l'on avait un arbre binaire simple (pas BST)
           Procédons par le jumeau itératif du preorder récursif tout en s'arrêtant lorsque le noeud est trouvé
        """
        if self is None:
            return
        stack = [self.root]
        while len(stack) > 0 :
            node = stack.pop()
            if node.transaction_id == key_id :                      #Condition d'arrêt
                return node.transaction_amount
            else :
                if isinstance(node, Node) :
                    if node.right is not None:                      #Même remarque que dans la méthode << insert >>
                        stack.append(node.right)
                    if node.left is not None:
                        stack.append(node.left)
                """if  isinstance(node, BinarySearchTree) :     
                    if  node.root.right is not None:
                        stack.append(node.root.right)
                    if node.root.left is not None:
                        stack.append(node.root.left)"""
        return -1                                                   #Sinon

    def get_sub_plus_petit(self,node):
        """J'avais un moment trouvé utile de créer cette fonction mais finalement, je ne l'ai pas utilisée.
        Toutefois je vais la laisser puisqu'elle est compacte
        """
        while node.left:
            node = node.left
        return node
    def remove(self,node):
        """On s'inspire de nouveau du cours
        """
        res = self.root                 # On démarre de la racine
        while res.transaction_amount < node.transaction_amount:     #On boucle tant que l'on a pas atteint le noeud
            res = res.right                                         #A supprimer
        while res.transaction_amount > node.transaction_amount:
            res = res.right
        if res.right != None:                                       #Une fois atteint, on vérifie les mêmes conditions
            q = self.getNext(res)                                   #que dans le cours
            res.transaction_id = q.transaction_id
            res.transaction_amount = q.transaction_amount
            del q
        elif res.left != None:
            q = self.getPrevious(res)
            res.transaction_id = q.transaction_id
            res.transaction_amount = q.transaction_amount
            del q
        else :
            if res.father !=None:
                if res.father.left == res :
                    res.father.left = None
                else:
                    res.father.right = None

    def jemePlusImportanteUtil(self,root,j,tab  = [0] ) :
        """Il suffit de s'apercevoir que le parcours en inorder dans une BST renvoie les noeuds de manière triée
        (dans l'ordre croissant), ainsi j'utilise cette propriété en itérant d'abord sur le coté droit et puis seulement
        sur le coté gauche. Je m'assure ainsi d'avoir un algorithme compact de complexe O(n)
        """
        if isinstance(root, BinarySearchTree) :             #De nouveau chez moi le premier Noeud est un BinarySearchTree
            if root  :                                      #Puis il s'agit de Node. D'ou le is_isinstance
                self.jemePlusImportanteUtil(root.root.right,j)
                tab.append((root.root.transaction_id, root.root.transaction_amount))
                self.jemePlusImportanteUtil(root.root.left,j)
        if isinstance(root, Node) :
            if root  :
                self.jemePlusImportanteUtil(root.right,j)
                tab.append((root.transaction_id, root.transaction_amount))
                self.jemePlusImportanteUtil(root.left,j)
        return tab

    def jemePlusImportante(self,j):
        print("La", j, " eme plus grande valeur de transaction_amount dans cet arbre est,", self.jemePlusImportanteUtil(tree,3)[j][1], " qui a comme id" ,self.jemePlusImportanteUtil(tree,3)[j][0])
        return self.jemePlusImportanteUtil(tree, 3)[j]

    def jemePlusImportante_iterative(self,j, liste_iter = []):
        """Je remarque peu avant la remise que cette méthode ne marche pas à cause d'un défaillance au niveau du getNext
        dont je m'aperçois peu avant la remise, c'est bizarre j'avais pourtant bien testé le getNext
        Je prends le getNext de la racine puis le getNext du getNext et ainsi de suite jusqu'à tomber sur le jeme get next
        on aurait très bien pu faire une boucle for
        """
        tour = 0
        mini = self.getFirst()
        liste_iter.append((mini.transaction_id,mini.transaction_amount))
        next = self.getNext(mini)
        liste_iter.append((next.transaction_id,next.transaction_amount))
        while self.getNext(next) != -1 and tour < j  :
            next = self.getNext(next)
            liste_iter.append((mini.transaction_id,mini.transaction_amount))

        return liste_iter[j]

    def jemePlusImportanteUtil_fusion_util(self,root, tab=[]):
        """Réalise un parcours en inorder exclusivement pour la méthode fusion
        """
        if isinstance(root, BinarySearchTree):
            if root:
                self.jemePlusImportanteUtil_fusion_util(root.root.left)
                tab.append(root.root.transaction_amount)
                self.jemePlusImportanteUtil_fusion_util(root.root.right)
        if isinstance(root, Node):
            if root:
                self.jemePlusImportanteUtil_fusion_util(root.left)
                tab.append(root.transaction_amount)
                self.jemePlusImportanteUtil_fusion_util(root.right)
        return tab

    def jemePlusImportante_fusion(self,root):
        return self.jemePlusImportanteUtil_fusion_util(root)

    def fusion(self, tree_1, tree_2):
        sorted = self.jemePlusImportante_fusion(tree_1)                 #Je stocke un parcours inorder
        sorted1 = sorted[:]                                             #Je ne veux pas une copie profonde
        sorted2 = self.jemePlusImportante_fusion(tree_2)[len(sorted1):] #Je ne veux pas les éléments restés dans la liste
        merger = modified_mergeSort(sorted1, sorted2)                   #Je fais appel à une adaptation du tri mergesort
        return merger


#################################################################################################################################

#              Ci-dessous se trouvent les tests de fonctionnalité des méthodes de la classe BinarySearchTree modifiées

#################################################################################################################################

node1 = Node(transaction_id=1, transaction_amount=10)
node2 = Node(transaction_id=2, transaction_amount=20 )
node3 = Node(transaction_id=3, transaction_amount=30)
node4 = Node(transaction_id=4, transaction_amount=40 )
new_tree = BinarySearchTree(transaction_id=1, transaction_amount=10)
new_tree.insert(node2)
new_tree.insert(node3)
new_tree.insert(node4)
new_tree.remove(node3)
print(new_tree.find(4), "Found")
print(new_tree.getNext(node2).transaction_id, "get next ")
print(new_tree.getPrevious(node1), "get previous ")
def is_bst_non_rec(node, lower=-float('inf')):
    """ On retouche légèrement cette fonction du TP afin de s'arrurer que l'on a bel et bien un BST
    """
    if node is None:
        return True
    stack = []
    while len(stack) > 0 or node is not None:
        while node is not None:
            stack.append(node)
            if isinstance(node, BinarySearchTree) :
                node = node.root.left
            elif isinstance(node, Node) :
                node = node.left
        node = stack.pop()
        if node.transaction_amount < lower:
            return False
        lower = node.transaction_amount
        if isinstance(node, BinarySearchTree):
            node = node.root.left
        elif isinstance(node, Node):
            node = node.left
    return True
if is_bst_non_rec(new_tree) :
        print('Est-ce que notre arbre est un BST? ',  is_bst_non_rec(new_tree) )
                # PARCOURS EN NIVEAUU


#liste_de_tree = [BinarySearchTree(node3),BinarySearchTree(node1),BinarySearchTree(node4),BinarySearchTree(node2)]
#liste_de_tree = [BinarySearchTree(Node(160,1)),BinarySearchTree(Node(110,2)),BinarySearchTree(Node(140,3)),BinarySearchTree(Node(360,4))]
#liste_de_tree = [BinarySearchTree(Node(160,1)),BinarySearchTree(Node(110,2)),BinarySearchTree(Node(140,3)),BinarySearchTree(Node(360,4))]
l = [(1,160),(2,1160),(3 ,1420),(4 ,360)]
                # PARCOURS EN NIVEAUU

def preorder(tree):
    if (tree != None):
        print(tree.transaction_amount)
        preorder(tree.left)
        preorder(tree.right)


def slice(liste,i) :
    """Fonction qui slice une liste en deux sous liste à partir d'un index
    """
    liste_d = liste[:i]
    liste_g = liste[i + 1:] 	#Je veux strictement à gauche donc index +1
    return [liste_d, liste_g]

def recurCountTrees(T):
    """Fonction récursive principale de la question 1, elle sera le squelette de cette partie.
    Elle va créer chacun des BSTs et stocker leur racine respective dans une liste.
    L'on aura ensuite plus qu'à parcourir cette liste de racine et de l'afficher via une fonction
    <<ShowTree>> afin d'obtenir notre affichage
    """
    liste = [i for i in range(1, T + 1)]        #Je vais de 1 à N compris
    def _recur(liste):
        if  len(liste) == 0:					#Si pas de fils gauche ou droit
            return [None]  						#Le None doit être itérable d'ou les crochets
        res = []
        for i in range(len(liste)): 	  		#Pour index de la liste ordonée  : [1,2...N]
            slice_var = slice(liste,i)			#J'appelle la fonction <<slice>> définie plus haut
            droit = _recur(slice_var[0]) 	  	#Recursion : Je veux une sous liste gauche des éléments inférieurs
            gauche = _recur(slice_var[1]) 		#Recursion : Idem pour les supérieurs
            for g in gauche:			  		#Pour chaque fils gauche puis pour chaque fils droit
                for d in droit:
                    racine =BinarySearchTree( liste[i], 0)  		#Je crée une racine
                    if isinstance(racine, BinarySearchTree) :	#Puisque le premier objet est un BinarySearchTree
                        racine.root.left = g  			  		#J'assigne son fils gauche
                        racine.root.right = d  		  			#Idem pour son fils droit
                        res.append(racine)
                    elif isinstance(racine, Node):				#A partir du deuxieme objet, ce sont des objets Node
                        racine.left = g
                        racine.right = d
                        res.append(racine)
        return res												#Nos racines dans une liste
    return _recur(liste)
#recur_count_var = recurCountTrees(3)

_niveau_liste = []  #Cette liste est clé :  on appelera une fonction <<n_parcours_differents(count_tree)>>
                   #qui elle même appelera <<niveau(tree)>>
                   #Elle va contenir le codage des parcours de nos BSTs
def niveau(tree):
    """Parcours par niveau en utilisant une queue"""
    f = Queue()
    f.insert(tree)
    while not f.isEmpty():
        n = f.remove()
        if n != None:
            _niveau_liste.append((n.transaction_id))
            if isinstance(n, BinarySearchTree):
                f.insert(n.root.left)
                f.insert(n.root.right)
            elif isinstance(n, Node):
                f.insert(n.left)
                f.insert(n.right)

def n_parcours_differents(liste):
    """Cette fontion va coder dans une liste les N parcours structuraux possibles d'un BST via un parcours par niveau
      en utilisant la liste renvoyée par la fonction récursive <<_recurCountTrees>> et en appelant la fonction <<niveau>>
      sur chaque élément de la liste renvoyée par la fonction récursive __recurCountrees
      """
    for i in range(len(liste)):
        niveau(liste[i])
    return _niveau_liste

def ShowTree(liste_de_tree,decalage =2, root_space=10,space =10) :
    """Comme son nom l'indique, cette fonction imprime l'ensemble des arbres structuralement différents
       Je suis conscient qu'elle est un peu longue mais au moins, j'ai une impression correcte des arbres.
       Cette fonction prend en paramètre une liste de parcours i.e : [1,2,4,3]
    """
    i = 0
    while i < len(liste_de_tree)-1 :            		#J'impose cela afin de conditioner entre les indice i et i+2
            if i == 0 :
                print(" "*root_space,liste_de_tree[i] ) #Espace suivi de la racine

            #Si l'on a deux fils, il faut comparer l'élément en cours, avec ses deux successeurs
            #Il y a deux cas : soit successeur i+1 est plus grand et successeur i+2 est plus petit
            #ou bien vice versa
            if i < len(liste_de_tree)-2 and\
            (liste_de_tree[i+1] < liste_de_tree[i] < liste_de_tree[i+2] or
            (liste_de_tree[i+1] > liste_de_tree[i] >liste_de_tree[i+2])) :
                    print(" "*(space)+"/" ," "  + "\ " )
                    print(" "* (space-decalage) , liste_de_tree[i+1], " "* decalage, liste_de_tree[i+2])
                    i+=2  						#Si cette condition est remplie, on a comparé deux indices, donc notre
                                                #prochain indice courant sera incrémenté de 2
            #S l'on a seulement un fils gauche
            elif liste_de_tree[i]> liste_de_tree[i+1] :
                if liste_de_tree[i+1]> liste_de_tree[0] :
                    print(" "*(space)+ "/")
                    print(" " *(space-decalage), liste_de_tree[i+1])
                    space -= decalage
                    i+=1
                else :
                    print(" " * (space) + "/")
                    print(" " * (space - decalage), liste_de_tree[i + 1])
                    space -= decalage
                    i += 1
            # fils droit
            elif liste_de_tree[i] < liste_de_tree[i+1] :
                # les fils du niveau 0
                print(" " * (space+decalage) + "\ " )
                print((" " )*(space +decalage) , liste_de_tree[i+1])
                space += decalage
                i+=1
def subb_list(T, liste) :
    """Cette fonction a pour but de <<découper>> une liste en plusieurs sous liste (de mêmes tailles)
    et renvoie une liste contenant ces dernières
    Elle nous sera d'une grande aide pour imprimer les N arbres structuralement différents
    """
    sub_liste = liste
    ret = []
    while len(sub_liste) >= T:
        ret.append(sub_liste[:T])
        sub_liste = sub_liste[T:]
    return ret
#print(subb_list(4,_niveau_liste), "voici à quoi ressemble la découpe d'une sous liste")

def engendrer_n_structal_trees(T) :
    """Fonction qui va appeler la fonction <<recurCountTrees>> et utiliser ce qu'elle retourne pour
    print les N différents structuraux BSTs
    """
    count_tree = recurCountTrees(T)            	#On stocke la fonction récursive
    niveau_list = n_parcours_differents(count_tree) #On stocke l'ensemble des N différents parcours au sein d'une seule liste
    subb_list_var = subb_list(T,niveau_list)        #On stocke et découpe ces parcours en plusieurs sous-listes
    for i in range(len(subb_list_var)):             #Pour chaque sous-liste j'appelle ShowTree qui va imprimer un arbre selon le
        print(ShowTree(subb_list_var[i]))           #parcours codifié sous forme de liste
    Total_BST = len(subb_list_var)					#Le nombre de BSTs
    print("L'on trouve exactement", Total_BST,  "BSTs structuraux différents si T vaut", T)
    return  Total_BST

"""Entrer ici le nombre T souhaité : 
"""
print(engendrer_n_structal_trees(3))



#####################################################################################################################################
#                                        Partie 3 : jeme transaction
#                            Les deux méthodes demandées se trouvent en haut dans la classe
#####################################################################################################################################
"""Je crée les noeuds présents dans l'exercice 2 puis je crée l'arbre
"""
node1 = Node(transaction_id=5, transaction_amount=50 )
node2 = Node(transaction_id=3, transaction_amount=30 )
node3 = Node(transaction_id=7, transaction_amount=70)
node4 = Node(transaction_id=2, transaction_amount=20)
node5 = Node(transaction_id=4, transaction_amount=40)
node6 = Node(transaction_id=6, transaction_amount=60)
node7 = Node(transaction_id=80, transaction_amount=80)
tree = BinarySearchTree(node1.transaction_id,node1.transaction_amount)
tree.insert(node2)
tree.insert(node3)
tree.insert(node4)
tree.insert(node5)
tree.insert(node6)
tree.insert(node7)

"""Entrer ici la jeme plus grande valeur souhaitée :
"""
print(tree.jemePlusImportante(3))
#print(tree.jemePlusImportante_iterative(2), "version itérative ")
#print(tree.getNext(node7).transaction_id," next ")
def jemePlusImportanteUtil_dans_lordre(root, tab=[]):
    """Cette fonction est simplement un parcours inorder, en constraste avec sa soeur inorder inversé"""
    if root:
        if isinstance(root, BinarySearchTree) :
            jemePlusImportanteUtil_dans_lordre(root.root.left)
            tab.append((root.transaction_id, root.root.transaction_amount))
            jemePlusImportanteUtil_dans_lordre(root.root.right)
        elif isinstance(root,Node) :
            jemePlusImportanteUtil_dans_lordre(root.left)
            tab.append((root.transaction_id, root.transaction_amount))
            jemePlusImportanteUtil_dans_lordre(root.right)
    return tab


#############################################################################################################################
#                                   """"PARTIE 4  du fichier  : Transactions"""
#############################################################################################################################

"""La même liste à transformer que dans l'exemple : """
liste_a_transformer_init = [(1,160),(2,1160),(3 ,1420),(4 ,360),(5 ,620),(6 ,640),(7 ,1260),(8 ,1840),(9 ,860),(10 ,60),(11 ,1820),(12 ,1860),(13 ,500),(14 ,1600),(15 ,560)]

""""J'initialise la racine de l'arbre comme le premier tuple de <<liste_a_transformer>>"""
correction_tree = BinarySearchTree(liste_a_transformer_init[0][0], liste_a_transformer_init[0][1])

"""Je vais ajouter comme noeud au BST <<correction_tree>> chaque élement de la <<liste_a_transformer ( à partir du deuxieme élément)"""
for i in range(1,len(liste_a_transformer_init)):
    correction_tree.insert((Node(liste_a_transformer_init[i][0],liste_a_transformer_init[i][1])))

def convert_tuple_to_node(tuple) :
    """Cette fonction converti un tuple en un noeud
    sauf si le tuple vaudra _1
    """
    if tuple == -1 :
        return -1
    return Node(tuple[0],tuple[1])

def convert_node_to_tuple(node) :
    """Idem mais inversement
    """
    if node == -1 :
        return -1
    return (node.transaction_id,node.transaction_amount)

def convert_node_to_tuple_ordered(node,i) :
    """Fait la même chose que <<convert_node_to_tuple_ordered>> mais assigne -1 au montant de transaction
     comme dans l'affichage de l'exemple
     """
    if node == -1 :
        return (i,-1)
    return (i,node.transaction_amount)

def cible(root,node_cible) :
    """Cette fonction est définie sur une liste ordonnée, qui est elle même définie par le parcours en inorder d'un BST.
       Cette fonction NE S OCCUPE QUE des POSITIONS PHYSIQUES DANS L'ARBRE et NON PAS des positions dans la liste à tranformer!
       Il est à noter que  :
       La position du suivant dans un parcours inorder vaut -1 si elle n'existe pas.
       """
    cible = convert_node_to_tuple(node_cible)               #Je converti  le noeud en un en tuple
    liste = jemePlusImportanteUtil_dans_lordre(root)        #Je crée une liste ordonée grâce à un parcourt en inorder
    index = liste.index(cible)                              #Je stocke l'index de l'élément duquel on veut prendre le suivant
    if index == len(liste)-1 :                              #Trivial : Si je me retrouve en fin de liste, il n'y a plus de suivant
        ret = -1
    else :
        ret = convert_tuple_to_node(liste[index+1])         #S'il y a un suivant je le retourne, il est en indice +1 dans la liste
    return ret
def next_node(start : Node, liste, tree) :
    """Cette fonction prend comme point de départ un noeud et va itérer jusque la fin de la liste
     afin de vérifier s'il existe un noeud de valeur plus grande, le cas échéant elle renvoie -1.
     Sinon elle renvoie donc l'élément suivant en appelant la fonction <<cible>>
     Elle nous fait gagner un temps précieux, car au lieu de chercher le suivant de start, puis se rendre compte qu'il n'est pas dans
     les éléments de la liste qui le succèdent, et donc chercher de nouveau le suivant, etc... jusqu'à se rendre compte
     qu'il n'y a pas de suivant ; on peut directement renvoyé -1
     """
    count = 0
    if start == -1:
        return
    for i in range(1, len(liste)):
        if start.transaction_amount > liste[i][1]:            #Tant qu'on ne trouve pas un élément plus grand
            count += 1
    if count == len(liste) - 1:                               #Si start est plus grand que tout le reste de la liste
                return -1
    return cible(tree, start)

def str_to_tuple(a_str : str) :
    """Transforme un str du fichier transactions.txt par un tuple
    """
    replace_1 = a_str.replace('(','')
    replace_2 = replace_1.replace(")", "")
    my_final_resulat =  eval(replace_2)
    return my_final_resulat

def open_file() :
    """Comme son nom l'indique, cette fonction ouvre le fichier transaction et va ajouter dans une liste
    les elemetn ligne par ligne du fichier corrected
    """
    liste = []
    with open('transactions.txt', 'r') as list :
        for line in list :
            l = line[:-1]
            liste.append(l)
    return liste
def transform_list_of_str_to_list_of_tuple(liste) :
    """Va transformer chaque élément d'un liste de str en un tupe en appelant la fonction <<str_to_tuple>>
    """
    liste = [str_to_tuple(liste[i]) for i in range(len(liste))]
    return liste

""" Les transactions du fichier transactions dans une liste """
liste_a_transformer_priorityBST_recur= transform_list_of_str_to_list_of_tuple(open_file())


def priorityBST_recur(liste,node : Node ,tree, ret = [], i =0) :
    """Cette fonction doit itérer sur la liste à transformer et renvoyer pour chaque élément de cette liste
    son élément directement plus grand en usant de la fonction <<cible>> prédéfinie ci-dessus
    """
    next = next_node(node,liste, tree)
    if len(liste) == 1 :                                #Trivial
        ret.append(convert_node_to_tuple_ordered(-1,i))
        return ret
    elif next == -1 :                                   #Pas de suivant plus grand
        ret.append(convert_node_to_tuple_ordered(-1,i))
    elif convert_node_to_tuple(next) in liste[liste.index(convert_node_to_tuple(node)):]:   #On a un suivant plus grand
        ret.append(convert_node_to_tuple_ordered(next,i))
    else :                           #Le suivant directement plus grand ne figure pas dans le reste de la liste
        while convert_node_to_tuple(next) not in liste[liste.index(convert_node_to_tuple(node)):] : #Tant qu'il n'y figure pas
            next = next_node(next,liste,tree)                                                       #On reprend le prochain
        ret.append(convert_node_to_tuple_ordered(next,i))                                           #Si trouvé, on l'ajoute
    return  priorityBST_recur(liste[1:],convert_tuple_to_node(liste[1:][0]),tree,ret, i+1)          #On refait la même chose
                                                                                                    #récursivement pour le reste
                                                                                                    #de la liste
print(priorityBST_recur(liste_a_transformer_priorityBST_recur,convert_tuple_to_node(liste_a_transformer_priorityBST_recur[0]),correction_tree, ret = [], i=1) )

"""La liste de l'exemple  :   """
liste_1 = [(1 ,360),(2 ,1260),(3 ,1600),(4 ,500),(5 ,640), (6 ,860),(7 ,1600),(8 ,1860),(9 ,1600),(10 ,500),(11 ,1860),(12 , -1),(13 ,560),(14 , -1),(15 , -1)]

"""La liste que j'obtiens : """
liste_2 = [(1, 360), (2, 1260), (3, 1600), (4, 500), (5, 640), (6, 860), (7, 1600), (8, 1860), (9, 1600), (10, 500), (11, 1860), (12, -1), (13, 560), (14, -1), (15, -1)]

"""Vérification  :   """
print(liste_1 == liste_2, "j'obtiens la liste souhaitée")

def write_file() :
    """Cette fonction va se charger d'écrire dans le fichier transactions_X__corrected.txt"""
    liste =priorityBST_recur(liste_a_transformer_priorityBST_recur,convert_tuple_to_node(liste_a_transformer_priorityBST_recur[0]),correction_tree, ret = [], i=1)
    liste = [ str(elem) for elem in liste]
    file2 = open('transactions_X_corrected.txt','w')
    file2.write('\n'.join(liste))
#print(write_file())


####################################################################################################################################
#                                                       Partie 5 Fusion
######################################################################################################################################


def modified_mergeSort(alist,blist):
    """On va se contenter de jouir du fait que les deux listes recues en paramètre sont déjà triées
    et, adapter le tri fusion du cours, en n'utilisant seulement que la partie du code relevante
    """
    n = len(alist)
    m = len(blist)
    i = 0
    j = 0
    k = 0
    ordered_liste=[0 for i in range(n+m)]
    while i< n and j < m  :
        if alist[i] < blist[j]  :
            ordered_liste[k] = alist[i]
            i = i + 1
        else :
            ordered_liste[k] = blist[j]
            j+=1
        k = k+1
    return ordered_liste[:k] + alist[i:] + blist[j:]



"""Les deux arbres comme dans l'exemple 2"""
root1 = BinarySearchTree(transaction_id=1, transaction_amount=16)
node100 = Node(1,4)
node101 = Node(1,20)
node102 = Node(1,2)
root1.insert(node100)
root1.insert(node101)
root1.insert(node102)

root2 = BinarySearchTree(transaction_id=2, transaction_amount=15)
node201 = Node(2,9)
node202 = Node(2,0)
root2.insert(node201)
root2.insert(node202)

a = BinarySearchTree(0,0).fusion(root1,root2)

"""On a bien le résultat voulu """
print("Resultat de la fusion :" , end=" ")
for node in a :
    print(str(node), end = " ")

