import java.util.BitSet;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        var solutionOne = new FirstSolution();
        var dailySolution = new DailySolutionOne();
        int[] nums = new int[]{3,0,1};
        System.out.println(solutionOne.missingNumber(nums));
    }
}