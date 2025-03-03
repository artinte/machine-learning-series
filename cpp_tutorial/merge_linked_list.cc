#include <iostream>
#include <vector>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// convert a vector to a singly linked list
ListNode* vectorToLinkedList(const std::vector<int>& nums) {
    if (nums.empty())
        return nullptr;
    
    ListNode* head = new ListNode(nums[0]);
    ListNode* current = head;

    for (size_t i = 1; i < nums.size(); i++) {
        current->next = new ListNode(nums[i]);
        current = current->next;
    }

    return head;
}

// merge two sorted linked lists (iterative approach)
ListNode* mergeTwoList(ListNode* l1, ListNode* l2) {
    // dummy node to simplify operations
    ListNode dummy(0);
    ListNode* tail = &dummy;

    while (l1 && l2) {
        if (l1->val < l2->val) {
            tail->next = l1;
            l1 = l1->next;
        } else {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;
    }

    if (l1) {
        tail->next = l1;
    }
    if (l2) {
        tail->next = l2;
    }

    return dummy.next;
}

void printList(ListNode* head) {
    while (head) {
        std::cout << head->val << " -> ";
        head = head->next;
    }
    std::cout << "nullptr" << std::endl;
}

void deleteList(ListNode* head) {
    while (head) {
        ListNode* temp = head;
        head = head->next;
        delete temp;
    }
}

int main() {
    std::vector<int> nums1 = {1, 3, 5, 8, 9};
    std::vector<int> nums2 = {2, 4, 6,};

    ListNode* l1 = vectorToLinkedList(nums1);
    ListNode* l2 = vectorToLinkedList(nums2);

    printList(l1);
    printList(l2);

    ListNode* merged = mergeTwoList(l1, l2);
    printList(merged);

    deleteList(merged);
    return EXIT_SUCCESS;
}
