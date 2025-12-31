import java.util.ArrayList;
import java.util.List;

public class InterviewSolution {
    /** <i>Google</i> */
    // 792. Number of Matching Subsequences
    /** Arrays & Hashing */
    public int numMatchingSubseq(String s, String[] words) {
        List<Integer>[] freq = new List[26];
        for(int i=0;i<26;i++) {
            freq[i] = new ArrayList<>();
        }
        for(int i=0;i<s.length();i++){
            freq[s.charAt(i) - 'a'].add(i);
        }
        int res = 0;
        for(var word : words){
            int lastIndex = -1;
            int[] f = new int[26];
            boolean confirm = true;
            for(var c : word.toCharArray()){
                int i = c-'a';
                int l = 0;
                int r = freq[i].size() - 1;
                int index = -1;
                while (l <= r){
                    int mid = (l+r)/2;
                    if(freq[i].get(mid) > lastIndex){
                        index = freq[i].get(mid);
                        r = mid - 1;
                    }
                    else{
                        l = mid + 1;
                    }
                }
                if(index > lastIndex){
                    lastIndex = index;
                }
                else{
                    confirm = false;
                    break;
                }
            }
            if(confirm) res += 1;
        }
        return res;
    }
    // 2351. First Letter To Appear Twice
    /** Count Frequency */
    public char repeatedCharacter(String s) {
        int[] freq = new int[26];
        for(var c : s.toCharArray()){
            if(freq[c - 'a'] > 0) return c;
            freq[c - 'a'] += 1;
        }
        return ' ';
    }
}
