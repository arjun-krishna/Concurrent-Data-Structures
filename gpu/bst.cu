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
	else if (root->data < key) return find(root->left, key);
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
	if (root == NULL) return NULL;

	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

__device__ void delete(node* root, int key) {
	if (root == NULL) return NULL;

	lock(root);

	node* node2delete = find(root, key);

	if (node2delete) {
		node* parent = node2delete->parent;
		if (parent) {
			lock(parent);
			unlock(root);
			node* successor = min_BST(node2delete);
			if (successor == NULL) {										// Leaf Node
				if (node2delete->data < parent->data) {
					parent->left = NULL;
				} else {
					parent->right = NULL;
				}
				free(node2delete);
			} 
			else if (successor != NULL) {
				node* parent_of_successor = successor->parent;
				node2delete->data = successor->data;
				if (successor->data < parent_of_successor->data) {
					parent_of_successor->left = NULL;
				} else {
					parent_of_successor->right = NULL;
				}
				free(successor);
			}
			unlock(parent);
		} else {												// ROOT of tree involved!
			// not handled!
		}
	} else {
		unlock(root);
	}
}



	// if (root->data == key) {  // directly the node to be deleted!
	// 	// problematic
	// }

	// else {
	// 	if (root->left && root->left->data == key) {
	// 		lock(root);
	// 	} 
	// 	else if (root->right && root->right->data == key) {
	// 		lock(root);
	// 	} 
	// 	else if (root->left && root->data > key) {

	// 		root->left = delete(root->left, key);
	// 	} 

	// }





	// int acquired = lock(root);

	// if (acquired) {

	// 	if (root->data == key) {				// The node to delete 
	// 		if (root->left == NULL) {
	// 			node* tmp = root->right;
	// 			free(root);
	// 			return tmp;
	// 		}
	// 	} else if (key < root->data) {
	// 		unlock(root);
	// 		root->left = delete(root->left, key);
	// 	}
	// 	else {
	// 		unlock(root);
	// 		root->right = delete(root->right, key);
	// 	}
	// } else {
	// 	root = delete(root, key);
	// }

	// if (key < root->data) 
	// 	root->left  = delete(root->left, key);
	// else if (key > root->data) 
	// 	root->right = delete(root->right, key); 
	// else {
	// 	if (root->left == NULL) {
	// 		node* tmp = root->right;
	// 		free(root);
	// 		return tmp;
	// 	} 
	// 	else if (root->right == NULL) {
	// 		node* tmp = root->left;
	// 		free(root);
	// 		return tmp;
	// 	}
	// 	// successor
	// 	node *tmp = min_BST(root->right);
	// 	root->data = tmp->data;
	// 	root->right = delete(root->right, tmp->data);
	// }
	// return root;


