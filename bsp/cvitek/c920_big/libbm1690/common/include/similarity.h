#include <algorithm>

template <typename T>
float max_err(const T *p_exp, const T *p_got, int len)
{
  float max_val = 0.0f;
  for (int idx = 0; idx < len; idx++) {
    max_val = std::max<float>(fabs(p_exp[idx] - p_got[idx]), max_val);
  }
  return max_val;
}

template <typename T>
float avg_err(const T *p_exp, const T *p_got, int len) {
    float sum = 0.0f;
    for (int idx = 0; idx < len; idx++) {
      sum += fabs(p_exp[idx] - p_got[idx]);
    }
    return sum / len;
}

template <typename T>
float dot(const T *v1, const T *v2, int len) {
  float result = 0.0f;
  for (int idx = 0; idx < len; idx++) {
    result += v1[idx] * v2[idx];
  }
  return result;
}

template <typename T>
float l2_norm(const T *v, int len) {
  float sum = 0.0f;
  for (int idx = 0; idx < len; idx++) {
    sum += v[idx] * v[idx];
  }
  return sqrtf(sum);
}

// from pytorch
template <typename T>
float cos_dist(const T *p_exp, const T *p_got, int len) {
  const float eps = 1e-8;
  float dot_prod = dot(p_exp, p_got, len);
  float norm_exp = l2_norm(p_exp, len);
  float norm_got = l2_norm(p_got, len);
  if (norm_exp == 0.f && norm_got == 0.f) {
    return 1.0;
  }
  float norm_prod = norm_exp * norm_got;
  float deno = std::max(norm_prod, eps);
  return dot_prod / deno;
}

template <typename T>
double l2_dist(const T *p_exp, const T *p_got, int len) {
  double sum = 0.0f;
  for (int idx = 0; idx < len; idx++) {
    double diff = p_exp[idx] - p_got[idx];
    sum += diff * diff;
  }
  return sqrt(sum);
}

template <class T>
float L2_similarity(const T *p_exp, const T *p_got, int len)
{
    double L2_dis = l2_dist(p_exp, p_got, len);
    double sum = 0.0f;
    for (int idx = 0; idx < len; idx++) {
    float mean = (p_exp[idx] + p_got[idx]) / 2.0f;
    sum += mean * mean;
    }
    ASSERT(sum > 0);
    return (float) (1 - L2_dis/std::sqrt(sum));
}

template <class T>
float ssd_cos_similiarity(const T *expect, const T *got, int len,int max,float rtol)
{
    const float eps = 1e-8;
    float dot_prod=0;
    float norm_exp=0;
    float norm_got=0;
    for (int i = 0; i < len/7; i++) {
        int index=i;
        for (int j = 0; j < 7; j++) {
            if (fabs(got[i*7+j] - expect[i*7+j]) > float(rtol)) {
                int same = 0;
                float min_score = 100.0;
                for (int k = -max; k < max; k++) {
                    if (fabs(got[(i+k)*7+j] - expect[(i+k)*7+j]) < float(rtol)) {
                        same = 1;
                        break;
                    }
                    if(!same){
                        if(fabs(got[(i+k)*7+j] - expect[(i+k)*7+j])<min_score){
                            min_score=fabs(got[(i+k)*7+j] - expect[(i+k)*7+j]);
                            index=i+k;
                        }
                    }
                }
            }
            dot_prod += got[index*7+j]*expect[index*7+j] ;
            norm_exp += expect[index*7+j]*expect[index*7+j];
            norm_got += got[index*7+j]*got[index*7+j];
        }
    }
    float norm_prod = sqrt(norm_exp) * sqrt(norm_got);
    float deno = norm_prod>eps?norm_prod:eps;
    return dot_prod / deno;
}


template <class T>
int do_similiarity_cmp_ssd(T *p_exp, T *p_got, int len,int max,float(rtol))
{
    int ret = 0;
    static float ssd_cos_tolerance = 0.8f;
    static float L2_tolerance = 0.9f;
    float max_error = max_err(p_exp, p_got, len);
    float avg_error = avg_err(p_exp, p_got, len);
    float cos_similiarity = ssd_cos_similiarity(p_exp, p_got, len,max,rtol);
    float L2_similiarity = L2_similarity(p_exp, p_got, len);
    printf("max_error %.20f \n", max_error);
    printf("avg_error  %.20f \n", avg_error);
    if(cos_similiarity >= ssd_cos_tolerance){
        printf("cos_similiarity= %.20f >= ssd_cos_tolerance = 0.8 OK\n",cos_similiarity);
    }else{
        printf("cos_similiarity= %.20f < ssd_cos_tolerance = 0.8 Failure\n",cos_similiarity);
    }
     if(L2_similiarity >= L2_tolerance){
        printf("L2_similiarity= %.20f >= L2_tolerance = 0.9 OK\n",L2_similiarity);
    }else{
        printf("L2_similiarity= %.20f < L2_tolerance = 0.9 Failure\n",L2_similiarity);
    }
    if(cos_similiarity < ssd_cos_tolerance
        || L2_similiarity < L2_tolerance)
        ret = -1;

    return ret;
}

template <class T>
int do_similiarity_cmp(T *p_exp, T *p_got, int len)
{
    int ret = 0;
    static float cos_tolerance = 0.98f;
    static float L2_tolerance = 0.9f;
    float max_error = max_err(p_exp, p_got, len);
    float avg_error = avg_err(p_exp, p_got, len);
    float cos_similiarity = cos_dist(p_exp, p_got, len);
    float L2_similiarity = L2_similarity(p_exp, p_got, len);
    printf("max_error %.20f \n", max_error);
    printf("avg_error  %.20f \n", avg_error);
    if(cos_similiarity >= cos_tolerance){
        printf("cos_similiarity= %.20f >= cos_tolerance = 0.98 OK\n",cos_similiarity);
    }else{
        printf("cos_similiarity= %.20f < cos_tolerance = 0.98 Failure\n",cos_similiarity);
    }
     if(L2_similiarity >= L2_tolerance){
        printf("L2_similiarity= %.20f >= L2_tolerance = 0.9 OK\n",L2_similiarity);
    }else{
        printf("L2_similiarity= %.20f < L2_tolerance = 0.9 Failure\n",L2_similiarity);
    }
    if(cos_similiarity < cos_tolerance
        || L2_similiarity < L2_tolerance)
        ret = -1;

    return ret;
}
