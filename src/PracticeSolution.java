import java.util.ArrayDeque;
import java.util.Deque;

public class PracticeSolution {
    // 8. String to Integer (atoi)
    /** String */
    public int myAtoi(String s) {
        int sign = 1;
        int res = 0;
        boolean firstCharChecked = false;
        int MAX_VALUE = Integer.MAX_VALUE;
        int MIN_VALUE = Integer.MIN_VALUE;
        for(var c : s.toCharArray()){
            if(c == ' '){
                if(firstCharChecked) break;
                else continue;
            }
            if(c == '+' || c == '-'){
                if(firstCharChecked) break;
                else{
                    sign = c == '+' ? 1 : -1;
                    firstCharChecked = true;
                    continue;
                }
            }
            if(!firstCharChecked) firstCharChecked = true;
            int number = c - '0';
            if (number < 0 || number > 9) break;
            if (sign == 1 && (MAX_VALUE - number) / 10 < res) {
                res = MAX_VALUE;
                break;
            }
            else if (sign == -1 && res < (MIN_VALUE + number) / 10){
                res = MIN_VALUE;
                break;
            }
            else {
                res = res * 10 + sign * number;
            }
        }
        return res;
    }
    // 9. Palindrome Number
    /** String */
    public boolean isPalindrome(int x) {
        var str = String.valueOf(x);
        int length = str.length();
        int left = 0;
        int right = length-1;
        while (left < right){
            if(str.charAt(left) != str.charAt(right)) return false;
            left += 1;
            right -= 1;
        }
        return true;
    }
}
