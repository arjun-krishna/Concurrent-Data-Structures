#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct node {
	int data;
	struct node *left;
	struct node *right;
	int sema;
} node;

void lock(node* n) {
	do {
		old = atomicCAS(&n->sema, 0, 1);
	} while (old == 1);
}

void unlock(node* n) {
	n->sema = 0;
}

__device__ node* new_node(int val) {
	node *tmp = (node *) malloc(sizeof(node));
	tmp->data = val;
	tmp->left = tmp->right = NULL;
	tmp->lock = 0;
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
		root = new_node(key); 
		return;
	}

	lock(root);

	if (key < root->data) {
		if (root->left == NULL) {			// Can be inserted to the immediate left
			root->left = new_node(key);
			unlock(root);
			return;
		} else {											// Release this Node and proceed
			unlock(root);
			insert(root->left, key);
		}
	} else {
		if (root->right == NULL) {		// Can be inserted to the immediate right
			root->right = new_node(key);
			unlock(root);
			return;
		} else {
			unlock(root);								// Release this Node and proceed
			insert(root->right, key);
		}
	}
}

__device__ node* min_BST(node* root) {
	if (root == NULL) return NULL;
	node* tmp = root;
	while(tmp->left != NULL)	tmp = tmp->left;
	return tmp;
}

__device__ void delete(node* root, int key) {
	if (root == NULL) return;

	lock(root);

	if (root->data == key) {				// The node to delete 
		
	} else if (key < root->data) {
		unlock(root);
		delete(root->left, key);
	}
	else {
		unlock(root);
		delete(root->right, key);
	}

	if (key < root->data) 
		root->left  = delete(root->left, key);
	else if (key > root->data) 
		root->right = delete(root->right, key); 
	else {
		if (root->left == NULL) {
			node* tmp = root->right;
			free(root);
			return tmp;
		} 
		else if (root->right == NULL) {
			node* tmp = root->left;
			free(root);
			return tmp;
		}
		// successor
		node *tmp = min_BST(root->right);
		root->data = tmp->data;
		root->right = delete(root->right, tmp->data);
	}
	return root;
}


