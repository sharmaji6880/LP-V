#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <omp.h>

using namespace std;

class Node {
    public:
        int data;
        Node *left,*right;
};

Node* insert(Node *root, int data) {
    if(root == NULL) {
        root = new Node;
        root->data = data;
        root->left = NULL;
        root->right = NULL;
        return root;
    }
    queue<Node*> q;
    q.push(root);

    while(!q.empty()) {
        Node *current = q.front();
        q.pop();

        if(current->left == NULL) {
            Node *node = new Node;
            node->data = data;
            node->left = NULL;
            node->right = NULL;
            current->left = node;
            return root;
        }
        else {
            q.push(current->left);
        }

        if(current->right == NULL) {
            Node *node = new Node;
            node->data = data;
            node->left = NULL;
            node->right = NULL;
            current->right = node;
            return root;
        }
        else {
            q.push(current->right);
        }
    }
    return root;
}

void dfs(Node *root) {
    stack<Node*> st;
    st.push(root);

    while(!st.empty()) {
        Node *top;
        #pragma omp critical 
        {
            top = st.top();
            st.pop();
        }

        cout<<top->data<<" ";

        if(top -> right) {
            #pragma omp critical 
            {
                st.push(top->right);
            }
        }
        if(top -> left) {
            #pragma omp critical 
            {
                st.push(top->left);
            }
        }
    }
}

void bfs(Node *root) {
    queue<Node*> q;
    int size;
    q.push(root);
    while(!q.empty()) {
        size = q.size();
        #pragma omp parallel for
        for(int i=0;i<size;i++) {
            Node *front;
            #pragma omp critical
            {
                front = q.front();
                q.pop();
            }
            cout<<front->data<<" ";

            if(front->left) {
                #pragma omp critical
                {
                    q.push(front->left);
                }
            }
            if(front->right) {
                #pragma omp critical
                {
                    q.push(front->right);
                }
            }
        }
    }
}

int main() {
    int n;
    cout<<"Enter size of array:";
    cin>>n;
    vector<int> arr(n);
    cout<<"Enter elements to be inserted in the tree:\n";
    for(int i=0;i<n;i++) {
        cin>>arr[i];
    }
    Node *root = NULL;

    for(int i=0;i<n;i++) {
        root = insert(root,arr[i]);
    }
    double start_time,end_time;

    // For Depth First Search
    cout<<"Depth First search is: ";
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            dfs(root);
        }
    }
    end_time = omp_get_wtime();
    cout<<"\n";
    cout<<"Time taken for dfs is:"<<end_time-start_time<<endl;

    // For Breadth First Search
    start_time = omp_get_wtime();
    cout<<"Breadth First Search is: ";
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            bfs(root);
        }
    }
    end_time = omp_get_wtime();
    cout<<"\n";
    cout<<"Time taken for bfs is:"<<end_time-start_time<<endl;
    
    
    return 0;

}