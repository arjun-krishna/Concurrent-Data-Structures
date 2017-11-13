#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

typedef struct node {
  int data;
  struct node *parent;
  struct node *left;
  struct node *right;
  int sema;
  int height;
} node;

__device__ node* global_Root;

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
  tmp->height = 1;
  return tmp;
}

__device__ int height(node *root)
{
    if (root == NULL)
        return 0;
    return root->height;
}

__device__ node* left_rotate(node* root,node* parent)
{
    node* temp1 = root->right;
    node* temp2 = temp1->left;
 
    temp1->left = root;
    root->parent = temp1;
    root->right = temp2;
    if(temp2)
      temp2->parent = root;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
  temp1->parent = parent;
    return temp1;
}

__device__ node* right_rotate(node* root, node* parent)
{
    node* temp1 = root->left;
    node* temp2 = temp1->right;
 
    temp1->right = root;
    root->parent = temp1;
    root->left = temp2;
    if(temp2)
      temp2->parent = root;
 
    root->height = max(height(root->left), height(root->right))+1;
    temp1->height = max(height(temp1->left), height(temp1->right))+1;
 
  temp1->parent = parent;
    return temp1;
}

__device__ int get_balance(node *root)
{
    if (root == NULL)
        return 0;
    return height(root->left) - height(root->right);
}

__device__ int MASTER_LOCK = 0;

__device__ void coarse_rebalance(node* p, int key) {
  //printf("rebalance : %d\n", p->data);
  if (p->parent) {
    p->height = max(height(p->left), height(p->right)) + 1;
    int balance = get_balance(p);

    bool rebalancing_occured = false;
    if (balance > 1 && key < p->left->data) {
      node* parent=p->parent;
      if (p->data < p->parent->data) {
        parent->left = right_rotate(p, p->parent);
      } else {
        parent->right = right_rotate(p, p->parent);
      }
      rebalancing_occured = true;
    }

    // Right Right Case
    else if (balance < -1 && key > p->right->data) {

      node* parent=p->parent;
      if (p->data < p->parent->data) {
        parent->left = left_rotate(p, p->parent);
      } else {
        parent->right = left_rotate(p, p->parent);
      }
      rebalancing_occured = true;
    }

    // Left Right Case
    else if (balance > 1 && key > p->left->data)
    {
      p->left =  left_rotate(p->left, p);
      node* parent=p->parent;
      if (p->data < p->parent->data) {
        parent->left = right_rotate(p, p->parent);
      } else {
        parent->right = right_rotate(p, p->parent);
      }
    rebalancing_occured = true;
    }

    // Right Left Case
    else if (balance < -1 && key < p->right->data)
    {
      p->right = right_rotate(p->right, p);
      node* parent=p->parent;
      if (p->data < p->parent->data) {
        parent->left = left_rotate(p, p->parent);
      } else {
        parent->right = left_rotate(p, p->parent);
      }
      rebalancing_occured = true;
    }

      if (!rebalancing_occured)
      coarse_rebalance(p->parent, key);
  } else {
    p->height = max(height(p->left), height(p->right)) + 1;
    int balance = get_balance(p);
    //printf("jag %d %d",balance,p->data);
    if (balance > 1 && key < p->left->data) {
      global_Root =  right_rotate(p, NULL);
    }

    // Right Right Case
    else if (balance < -1 && key > p->right->data) {
      global_Root = left_rotate(p, NULL);
  
    }

    // Left Right Case
    else if (balance > 1 && key > p->left->data)
    {
      p->left =  left_rotate(p->left, p);
      global_Root = right_rotate(p, NULL);
    }

    // Right Left Case
    else if (balance < -1 && key < p->right->data)
    {
      p->right = right_rotate(p->right, p);
      global_Root = left_rotate(p, NULL);
    }

  }
  return;
}


__device__ void coarse_insert(int key) {

  
  bool flag = true;
  while (flag) {
    if (!atomicExch(&MASTER_LOCK, 1)) {
      node* curr = global_Root;
      node* parent = NULL;
      while (curr != NULL) {  
        parent = curr;
        if (key < curr->data)
          curr = curr->left;
        else
          curr = curr->right; 
        if (curr == NULL) {
          if (key < parent->data) {
            parent->left = new_node(key, parent);
            coarse_rebalance(parent->left, key);
          } else {
            parent->right = new_node(key, parent);
            coarse_rebalance(parent->right, key);
          }     
        } else {
          if (parent)
            atomicExch(&(parent->sema), 0);
        }
      }
      flag = false;
      atomicExch(&MASTER_LOCK, 0);
    }
  }
  return; 
}

__device__ void coarse_delete(node* root, int key) {
  return;
}

__device__ node* find(node* root, int key) {
  if (root == NULL) return NULL;

  if (root->data == key) return root;
  else if (root->data > key) return find(root->left, key);
  else return find(root->right, key);
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
