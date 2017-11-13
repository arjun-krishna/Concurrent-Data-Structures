#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct node {
	int data;
	struct node *parent;
	struct node *left;
	struct node *right;
	int sema;
} node;

__device__ int lock(node* n) {
	return !atomicExch(&n->sema, 1);
}

__device__ void unlock(node* n) {
	atomicExch(&n->sema, 0);
}

__device__ node* new_node(int val, node* parent) {
	node *tmp = (node *) malloc(sizeof(node));
	tmp->data = val;
	tmp->parent = parent;
	tmp->left = tmp->right = NULL;
	tmp->sema = 0;
	return tmp;
}

__device__ node* find(node* root, int key) {
	if (root == NULL) return NULL;

	if (root->data == key) return root;
	else if (root->data > key) return find(root->left, key);
	else return find(root->right, key);
}


__device__ void insert(node* root, int key) {

	if (root == NULL) { 		 				// Empty Tree
		root = new_node(key, NULL); 
		return;
	}
	
	int acquired = lock(root);

	if (acquired) {
		if (key < root->data) {
			if (root->left == NULL) {			// Can be inserted to the immediate left
				root->left = new_node(key, root);
				unlock(root);
				return;
			} else {											// Release this Node and proceed
				unlock(root);
				insert(root->left, key);
			}
		} else {
			if (root->right == NULL) {		// Can be inserted to the immediate right
				root->right = new_node(key, root);
				unlock(root);
				return;
			} else {
				unlock(root);								// Release this Node and proceed
				insert(root->right, key);
			}
		}
	} else {
		insert(root, key);
	}
}


__device__ void pre_order(node* root)
{
    if(root != NULL)
    {
      printf("%d ", root->data);
      pre_order(root->left);
      pre_order(root->right);
    }
		return;
}

__device__ void in_order(node* root)
{
    if(root != NULL)
    {
      in_order(root->left);
      printf("%d ", root->data);
      in_order(root->right);
    }
		return;
}

__device__ node* min_BST(node* root) {
	node* tmp = root->right;
	if (tmp == NULL) return NULL;

	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

__device__ void bst_delete(node* root, int key) {
	if (root == NULL) return;
	// printf("del key= %d\n", key);
	//int root_acquired = lock(root);
	int root_acquired = 1;
	if (root_acquired) {
		node* node2delete = find(root, key);

		if (node2delete) {
			//printf("Delete Node %d\n",node2delete->data);
			node* parent = node2delete->parent;
			if (parent) {
				//unlock(root);
				int parent_acquired = lock(parent);
				if (parent_acquired) {
					node* successor = min_BST(node2delete);
					if (successor == NULL) {										// Leaf Node
						if (node2delete->data < parent->data) {
							parent->left = node2delete->left;
						} else {
							parent->right = node2delete->left;
						}
						if(node2delete->left)
							node2delete->left->parent = parent;
						free(node2delete);
					} 
					else if (successor != NULL) {
						node* parent_of_successor = successor->parent;
						node2delete->data = successor->data;
						if (successor->data < parent_of_successor->data) {
							parent_of_successor->left = successor->right;
						} else {
							parent_of_successor->right = successor->right;
						}
						if(successor->right)
							successor->right->parent = parent_of_successor;
						free(successor);
					}
					unlock(parent);
				} else {
					//printf("recall %d\n", key);
					bst_delete(root, key);
				}
			} else {												// ROOT of tree involved!
				// not handled!
			}
		} else {
			//unlock(root);
		}
	} else {
		//printf("recall %d\n", key);
		bst_delete(root, key);
	}
}