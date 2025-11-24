import java.util.List;

public class Main {
    public static void main(String[] args) {
        var solutionOne = new FirstSolution();
        String[] list = new String[]{"abbbbbbbbbbb","aaaaaaaaaaab"};
        System.out.println(solutionOne.groupAnagrams(list));
    }
}