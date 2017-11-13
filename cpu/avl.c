#include <stdio.h>
#include <stdlib.h>


typedef struct node {
  int data;
  struct node *left;
  struct node *right;
  int height;
} node;

int max(int a, int b)
{
    if(a > b)
      return a;
    return b;
}

node* new_node(int val) {
  node *tmp = (node *) malloc(sizeof(node));
  tmp->data = val;
  tmp->left = tmp->right = NULL;
  tmp->height = 1;
  return tmp;
}

int height(node *root)
{
    if (root == NULL)
        return 0;
    return root->height;
}

node* left_rotate(node* root)
{
    node* temp1 = root->right;
    node* temp2 = temp1->left;
 
    temp1->left = root;
    root->right = temp2;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

node* right_rotate(node* root)
{
    node* temp1 = root->left;
    node* temp2 = temp1->right;
 
    temp1->right = root;
    root->left = temp2;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
    return temp1;
}

int get_balance(node *root)
{
    if (root == NULL)
        return 0;
    return height(root->left) - height(root->right);
}

node* insert(node* root, int key) {

  //Normal BST key insertion
  if (root == NULL) return new_node(key);

  if (key < root->data) 
    root->left  = insert(root->left, key);
  else 
    root->right = insert(root->right, key);

  root->height = max(height(root->left),height(root->right)) + 1;
  int balance = get_balance(root);

  // Left Left Case
  if (balance > 1 && key < root->left->data)
    return right_rotate(root);

  // Right Right Case
  if (balance < -1 && key > root->right->data)
    return left_rotate(root);

  // Left Right Case
  if (balance > 1 && key > root->left->data)
  {
    root->left =  left_rotate(root->left);
    return right_rotate(root);
  }

  // Right Left Case
  if (balance < -1 && key < root->right->data)
  {
    root->right = right_rotate(root->right);
    return left_rotate(root);
  }

  //If Node is balanced
  return root;
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

node* min_BST(node* Node) {
  if (Node == NULL) return NULL;
  node* tmp = Node;
  while(tmp->left != NULL)  tmp = tmp->left;
  return tmp;
}

node* delete(node* root, int key) {
  if (root == NULL) return NULL;

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

  if (root == NULL)
      return root;

  root->height = max(height(root->left),height(root->right)) + 1;

  int balance = get_balance(root);

  // Left Left Case
  if (balance > 1 && get_balance(root->left) >= 0)
    return right_rotate(root);
 
  // Left Right Case
  if (balance > 1 && get_balance(root->left) < 0)
  {
    root->left =  left_rotate(root->left);
    return right_rotate(root);
  }
 
  // Right Right Case
  if (balance < -1 && get_balance(root->right) <= 0)
    return left_rotate(root);
 
  // Right Left Case
  if (balance < -1 && get_balance(root->right) > 0)
  {
    root->right = right_rotate(root->right);
    return left_rotate(root);
  }
 
  return root;
}