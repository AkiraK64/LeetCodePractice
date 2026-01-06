
/** <i>LinkedList</i> - 622. Design Circular Queue */
class MyCircularQueue {

    ListNode head;
    ListNode tail;
    int size;
    int maxSize;

    public MyCircularQueue(int k) {
        maxSize = k;
        head = null;
        tail = null;
        size = 0;
    }

    public boolean enQueue(int value) {
        if(size < maxSize){
            if(head == null){
                head = new ListNode(value);
            }
            else if(tail == null){
                tail = new ListNode(value);
                head.next = tail;
                tail.next = head;
            }
            else{
                var newNode = new ListNode(value);
                tail.next = newNode;
                newNode.next = head;
                tail = newNode;
            }
            size += 1;
            return true;
        }
        return false;
    }

    public boolean deQueue() {
        if(size == 0) return false;
        if(size == 1){
            head = null;
        }
        else if(size == 2){
            tail.next = null;
            head = tail;
            tail = null;
        }
        else{
            var curNode = head;
            head = head.next;
            tail.next = head;
            curNode.next = null;
        }
        size -= 1;
        return true;
    }

    public int Front() {
        if(head != null) return head.val;
        return -1;
    }

    public int Rear() {
        if(tail != null) return tail.val;
        if(head != null) return head.val;
        return -1;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public boolean isFull() {
        return size == maxSize;
    }
}