import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        var solutionOne = new SolutionOne();
        int[] nums = new int[]{1,0,0,0,1,0,0,1};
        System.out.println(solutionOne.kLengthApart(nums, 2));
    }
}