// From https://github.com/smartcorelib/smartcore/blob/main/src/algorithm/sort/quick_sort.rs
// Authors: Volodymyr Orlov
#![allow(clippy::many_single_char_names)]
pub trait QuickArgSort {
    fn quick_argsort_mut(&mut self) -> Vec<usize>;

    fn quick_argsort(&self) -> Vec<usize>;
}

impl QuickArgSort for Vec<f64> {
    fn quick_argsort(&self) -> Vec<usize> {
        let mut v = self.clone();
        v.quick_argsort_mut()
    }

    fn quick_argsort_mut(&mut self) -> Vec<usize> {
        let stack_size = 64;
        let mut jstack = -1;
        let mut l = 0;
        let mut istack = vec![0; stack_size];
        let mut ir = self.len() - 1;
        let mut index: Vec<usize> = (0..self.len()).collect();

        loop {
            if ir - l < 7 {
                for j in l + 1..=ir {
                    let a = self[j];
                    let b = index[j];
                    let mut i: i32 = (j - 1) as i32;
                    while i >= l as i32 {
                        if self[i as usize] <= a {
                            break;
                        }
                        self[(i + 1) as usize] = self[i as usize];
                        index[(i + 1) as usize] = index[i as usize];
                        i -= 1;
                    }
                    self[(i + 1) as usize] = a;
                    index[(i + 1) as usize] = b;
                }
                if jstack < 0 {
                    break;
                }
                ir = istack[jstack as usize];
                jstack -= 1;
                l = istack[jstack as usize];
                jstack -= 1;
            } else {
                let k = (l + ir) >> 1;
                self.swap(k, l + 1);
                index.swap(k, l + 1);
                if self[l] > self[ir] {
                    self.swap(l, ir);
                    index.swap(l, ir);
                }
                if self[l + 1] > self[ir] {
                    self.swap(l + 1, ir);
                    index.swap(l + 1, ir);
                }
                if self[l] > self[l + 1] {
                    self.swap(l, l + 1);
                    index.swap(l, l + 1);
                }
                let mut i = l + 1;
                let mut j = ir;
                let a = self[l + 1];
                let b = index[l + 1];
                loop {
                    loop {
                        i += 1;
                        if self[i] >= a {
                            break;
                        }
                    }
                    loop {
                        j -= 1;
                        if self[j] <= a {
                            break;
                        }
                    }
                    if j < i {
                        break;
                    }
                    self.swap(i, j);
                    index.swap(i, j);
                }
                self[l + 1] = self[j];
                self[j] = a;
                index[l + 1] = index[j];
                index[j] = b;
                jstack += 2;

                if jstack >= 64 {
                    panic!("stack size is too small.");
                }

                if ir - i + 1 >= j - l {
                    istack[jstack as usize] = ir;
                    istack[jstack as usize - 1] = i;
                    ir = j - 1;
                } else {
                    istack[jstack as usize] = j - 1;
                    istack[jstack as usize - 1] = l;
                    l = i;
                }
            }
        }

        index
    }
}
