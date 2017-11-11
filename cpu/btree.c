/* @auth : Arjun Krishna
 * @desc : Non-concurrent implementation of B-Tree 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct node {
	int *keys;											// Keys array
	int t;													// min degree
	struct node **child;						// pointer array for children
	int nkeys;											// Number of keys
	bool isLeaf;										// indicator of leaf
} node;

node* new_node(int _t, bool _isLeaf) {
	node *tmp = (node*) malloc(sizeof(node));
	tmp->t = _t;
	tmp->isLeaf = _isLeaf;
	tmp->keys = (int*) malloc(sizeof(int) * (2*_t-1));
	tmp->child = (node**) malloc(sizeof(node *) * (2*_t));
	tmp->nkeys = 0;
	return tmp;
}

// Splits node n, curr being the parent
void split_child(node* curr, int s, node* n) {
	int t = n->t;
	node *m = new_node(t, n->isLeaf);
	m->nkeys = t - 1;

	int i;
	for (i=0; i<(t-1); i++)
		m->keys[i] = n->keys[i+t];

	if (n->isLeaf == false) {
		for (i=0; i<t; i++)
			m->child[i] = n->child[i+t];
	}
	n->nkeys = t - 1;

	for (i=curr->nkeys; i>=(s+1); i--)
		curr->child[i+1] = curr->child[i];

	curr->child[s+1] = m;

	for (i=curr->nkeys-1; i>=s; i--)
		curr->keys[i+1] = curr->keys[i];

	curr->keys[s] = n->keys[t-1];
	curr->nkeys += 1;
}


void insert(node* root, int key, int _t) {
	if (root == NULL) {
		root = new_node(_t, true);
		root->keys[0] = key;
		root->nkeys = 1;
	} else {
		if (root->nkeys == 2*_t - 1) {						// Node is Full
			
			node *new_root = new_node(_t, false);
			new_root->child[0] = root;
			split_child(new_root, 0, root);

			int c = 0;
			if (new_root->keys[0] < key)
				c++;
			insert(new_root->child[c], key, _t);
		
		} else {																	// Node is Not Full

			int i = root->nkeys - 1;

			if (root->isLeaf) {
				while (i>=0 && root->keys[i] > key) {
					root->keys[i+1] = root->keys[i];
					i--;
				}
				root->keys[i+1] = key;
				root->nkeys += 1;
			} else {
				while (i>=0 && root->keys[i] > key) 
					i--;
				if (root->child[i+1]->nkeys == (2*_t-1)) {
					split_child(root, i+1, root->child[i+1]);
					if (root->keys[i+1] < key)
						i++;
				}
				insert(root->child[i+1], key, _t);
			}

		}
	}
}

node* find(node* root, int key) {
	int i = 0;
	while (i < root->nkeys && key > root->keys[i])
		i++;

	if (root->keys[i] == key)
		return root;

	if (root->isLeaf) return NULL;
	else return find(root->child[i], key);	
}

node* delete(node* root, int key) {
	if (root == NULL) return NULL;

	root->remove(k);

	if (root->nkeys == 0) {
		node *tmp = root;
		if (root->isLeaf) root = NULL;
		else root = root->child[0];
		free(tmp);
	}
}

void remove(node* root, int k)
{
    int idx = findKey(k);
 
    // The key to be removed is present in this node
    if (idx < n && keys[idx] == k)
    {
 
        // If the node is a leaf node - removeFromLeaf is called
        // Otherwise, removeFromNonLeaf function is called
        if (leaf)
            removeFromLeaf(idx);
        else
            removeFromNonLeaf(idx);
    }
    else
    {
 
        // If this node is a leaf node, then the key is not present in tree
        if (leaf)
        {
            cout << "The key "<< k <<" is does not exist in the tree\n";
            return;
        }
 
        // The key to be removed is present in the sub-tree rooted with this node
        // The flag indicates whether the key is present in the sub-tree rooted
        // with the last child of this node
        bool flag = ( (idx==n)? true : false );
 
        // If the child where the key is supposed to exist has less that t keys,
        // we fill that child
        if (C[idx]->n < t)
            fill(idx);
 
        // If the last child has been merged, it must have merged with the previous
        // child and so we recurse on the (idx-1)th child. Else, we recurse on the
        // (idx)th child which now has atleast t keys
        if (flag && idx > n)
            C[idx-1]->remove(k);
        else
            C[idx]->remove(k);
    }
    return;
}

 
int main()
{
    return 0;
}

 
