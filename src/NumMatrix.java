/** <i>Arrays & Hashing</i> - 304. Range Sum Query 2D - Immutable */
class NumMatrix {
    int[][] sum;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        sum = new int[m][n];
        int total = 0;
        for(int i=0;i<m;i++){
            total += matrix[i][0];
            sum[i][0] = total;
        }
        for(int j=1;j<n;j++){
            total = 0;
            for(int i=0;i<m;i++){
                total += matrix[i][j];
                sum[i][j] = sum[i][j-1] + total;
            }
        }
    }

    private int sumRectangle(int row, int col){
        if(row < 0 || col < 0) return 0;
        return sum[row][col];
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return sumRectangle(row2, col2) -
                (sumRectangle(row2, col1-1) + sumRectangle(row1-1, col2) -
                        sumRectangle(row1-1, col1-1));
    }
}