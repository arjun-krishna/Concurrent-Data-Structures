/* @auth : Arjun Krishna
 * @desc : Non-concurrent implementation of BST
 */

#include <stdio.h>
#include <stdlib.h>


typedef struct node {
  int data;
  struct node *left;
  struct node *right;
} node;


node* new_node(int val) {
  node *tmp = (node *) malloc(sizeof(node));
  tmp->data = val;
  tmp->left = tmp->right = NULL;
  return tmp;
}

node* insert(node* root, int key) {
  if (root == NULL) return new_node(key);

  if (key < root->data) 
    root->left  = insert(root->left, key);
  else 
    root->right = insert(root->right, key);
}

node* min_BST(node* root) {
  if (root == NULL) return NULL;
  node* tmp = root;
  while(tmp->left != NULL)  tmp = tmp->left;
  return tmp;
}

node* bst_delete(node* root, int key) {
  if (root == NULL) return NULL;

  if (key < root->data) 
    root->left  = bst_delete(root->left, key);
  else if (key > root->data) 
    root->right = bst_delete(root->right, key); 
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
    root->right = bst_delete(root->right, tmp->data);
  }
  return root;
}

node* find(node* root, int key) {
  if (root == NULL) return NULL;

  if (root->data == key) return root;
  else if (root->data < key) return find(root->left, key);
  else return find(root->right, key);
}

void in_order(node* root)
{
    if(root != NULL)
    {
      in_order(root->left);
      printf("%d ", root->data);
      in_order(root->right);
    }
    return;
}

void pre_order(node* root)
{
    if(root != NULL)
    {
        printf("%d ", root->data);
        pre_order(root->left);
        pre_order(root->right);
    }
    return;
}